import argparse
import asyncio
import json
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast,)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
PRINT_REQUESTS = 100
DEF_MAX_PROMPT_LEN = 1024
DEF_MAX_TOTAL_LEN = 2048

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


# trt llm not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.000001,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "min_length": request_func_input.output_len,
            "end_id": 1048576,
            **request_func_input.extra_request_body,
        }
        if args.disable_ignore_eos:
            del payload["min_length"]
            del payload["end_id"]
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "trt": async_request_trt_llm,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
    )

    return metrics, output_lens


async def benchmark(backend: str, api_url: str, model_id: str, tokenizer: PreTrainedTokenizerBase, input_requests: List[Tuple[str, int, int]], rate: float, disable_tqdm: bool, extra: Dict[str, Any],):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if len(input_requests) == 0:
        raise ValueError("Empty input requests")
    print("Starting initial single prompt test run...")
    prompt, prompt_len, output_len = input_requests[0]
    test_input = RequestFuncInput(model=model_id, prompt=prompt, api_url=api_url, prompt_len=prompt_len, output_len=output_len, extra_request_body=extra,)
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(f"Initial test run failed - Please make sure benchmark arguments are correctly specified. Error: {test_output.error}")
    else:
        print(f"Initial test run completed. Starting main benchmark run with rate: {rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for req in get_request(input_requests, rate):
        prompt, prompt_len, output_len = req
        request_func_input = RequestFuncInput(model=model_id, prompt=prompt, api_url=api_url, prompt_len=prompt_len, output_len=output_len, extra_request_body=extra,)
        tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, output_lens = calculate_metrics(input_requests=input_requests, outputs=outputs, dur_s=benchmark_duration, tokenizer=tokenizer, backend=backend,)

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", rate))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10}".format("Total generated tokens (retokenized):", metrics.total_output_retokenized))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    result = None
    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "timestampe": datetime.now().strftime("%m%d:%H-%M"),
            "backend": args.backend,
            "request_rate": rate,
            "num_prompts": args.num_prompts,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            #"generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
        }
    else:
        print(f"Error running benchmark for request rate: {rate}")
        print("-" * 30)

    # Append results to a JSONL file
    if args.output_file and result is not None:
        print(f"Append the resolts into file {args.output_file}")
        with open(args.output_file, "a") as file:
            file.write(json.dumps(result) + "\n")
    return result


def parse_request_rate_range(request_rate_range):
    if len(request_rate_range.split(",")) == 3:
        start, stop, step = map(int, request_rate_range.split(","))
        return list(range(start, stop, step))
    else:
        return list(map(int, request_rate_range.split(",")))


def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False

def download_sharegpt_dataset(path):
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    print(f"Downloading dataset from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        with open(path, "wb") as f, tqdm(desc="Downloading", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024,) as progress_bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)
        print(f"Dataset downloaded and saved to {path}")
    except requests.RequestException as e:
        raise Exception(f"Failed to download dataset: {e}")

def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra = {}
    if args.extra_request_body:
        extra = json.loads(args.extra_request_body)

    api_url = args.endpoint + "/completions"
    model_url = args.endpoint + "/models"
    backend = args.backend
    if backend == "trt":
        api_url = args.endpoint + "/models/ensemble/generate_stream"
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)

    # Get model name
    if args.model is None:
        try:
            response = requests.get(model_url)
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
            print(f"Got model name from server: {args.model}")
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    print(f"{args}\n")
    if args.tokenizer is None:
        print("Please input tokenizer path by --tokenizer")
        exit(1)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Read dataset
    if args.dataset not in ["sharegpt", "ppio"]:
        raise ValueError(f"Invalid dataset argument: {args.dataset}")
    if args.dataset == "ppio":
        dataset_path = os.path.dirname(__file__) + "/stability_samples.json"
    elif args.dataset == "sharegpt":
        dataset_path = os.path.dirname(__file__) + "/ShareGPT_V3_unfiltered_cleaned_split.json"
        if not os.path.exists(dataset_path):
            download_sharegpt_dataset(dataset_path)
    input_requests = []
    print(f"Load test data from {dataset_path}")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            json_data= json.load(f)
            if args.dataset == "ppio" and "kind" in json_data and json_data["kind"] == "ppio-internal":
                for i in reversed(range(len(json_data["data"]))):
                    data = json_data["data"][i]
                    if len(input_requests) >= args.num_prompts:
                        break
                    if args.fixed_input_len:
                        ids = tokenizer.encode(data["prompt"])
                        if args.fixed_input_len < data["prompt_len"]:
                            ids = ids[:args.fixed_input_len]
                        else:
                            ratio = (args.fixed_input_len + data["prompt_len"] - 1) // data["prompt_len"]
                            ids = (ids * ratio)[: args.fixed_input_len]
                        prompt = tokenizer.decode(ids)
                        input_requests.append((prompt, args.fixed_input_len, data["output_len"] if args.fixed_output_len is None else args.fixed_output_len))
                    elif data["prompt_len"] < 1024 and (args.fixed_output_len is None or data["prompt_len"] + data["output_len"] < 2048):
                        input_requests.append((data["prompt"], data["prompt_len"], data["output_len"] if args.fixed_output_len is None else args.fixed_output_len))
            elif args.dataset == "sharegpt":
                dataset = [data for data in json_data if len(data["conversations"]) >= 2]
                dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]
                random.shuffle(dataset)
                for i in range(len(dataset)):
                    if len(input_requests) >= args.num_prompts:
                        break
                    prompt = dataset[i][0]
                    prompt_token_ids = tokenizer.encode(prompt)
                    completion = dataset[i][1]
                    completion_token_ids = tokenizer.encode(completion)
                    prompt_len = len(prompt_token_ids)
                    output_len = (len(completion_token_ids) if args.fixed_output_len is None else args.fixed_output_len)
                    if prompt_len < 4 or output_len < 4:
                        continue
                    if args.fixed_input_len:
                        if args.fixed_input_len < prompt_len:
                            ids = prompt_token_ids[:args.fixed_input_len]
                        else:
                            ratio = (args.fixed_input_len + prompt_len - 1) // prompt_len
                            ids = (ids * ratio)[:args.fixed_input_len]
                        prompt = tokenizer.decode(ids)
                        input_requests.append((prompt, args.fixed_input_len, output_len))
                    else:
                        if prompt_len > 1024 or (prompt_len + output_len > 2048 and args.fixed_output_len is None):
                            continue
                        input_requests.append((prompt, prompt_len, output_len))
    if len(input_requests) == 0:
        raise RuntimeError(f"No data loaded from {dataset_path}")
    random.shuffle(input_requests)
    i = 0
    while args.num_prompts > len(input_requests):
        input_requests.append(input_requests[i])
        i += 1
    print(f"Load {len(input_requests)} requests")
    if PRINT_REQUESTS > 0:
        for i in range(min(len(input_requests), PRINT_REQUESTS)):
            req = input_requests[i]
            print(f"Request[{i}] prompt len: {len(tokenizer.encode(req[0]))}, {req[1]}, {req[2]}, {req[0][0:100]}")

    # Run benchmark
    if args.request_rate_range is not None and len(args.request_rate_range.split(",")) == 3:
        start, stop, step = map(int, args.request_rate_range.split(","))
        requests_rates = list(range(start, stop, step))
        for rate in requests_rates:
            asyncio.run(
                benchmark(backend=backend, api_url=api_url, model_id=args.model, tokenizer=tokenizer, input_requests=input_requests, rate=rate, disable_tqdm=args.disable_tqdm, extra=extra,))
    else:
        asyncio.run(
            benchmark(backend=backend, api_url=api_url, model_id=args.model, tokenizer=tokenizer, input_requests=input_requests, rate=args.request_rate, disable_tqdm=args.disable_tqdm, extra=extra,))
    print("DONE")

def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="vllm", help="Must specify a backend, depending on the LLM Inference Engine.",)
    parser.add_argument("--endpoint", type=str, help="Server or API base url if not using http host and port.",)
    parser.add_argument("--model", type=str, help="Name of the model. If not set, the default model will request /v1/models for conf.",)
    parser.add_argument("--tokenizer", type=str, help="Name or path of the tokenizer. If not set, using the model conf.",)
    parser.add_argument("--dataset", type=str, choices=list(["sharegpt", "ppio"]), default="ppio", help="The preset dataset name, can be: 'ppio', 'sharegpt'",)
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process. Default is 1000.",)
    parser.add_argument("--fixed-input-len", type=int, default=None, help="Input length for each request. Overrides the output length from the dataset.",)
    parser.add_argument("--fixed-output-len", type=int, default=None, help="Output length for each request. Overrides the output length from the dataset.",)
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",)
    parser.add_argument("--request-rate-range", type=str, default=None, help="Range of request rates in the format start,stop,step. for example: 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",)
    parser.add_argument("--seed", type=int, default=1, help="The random seed. Default is 1")
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.",)
    parser.add_argument("--disable-stream", action="store_true", help="Disable streaming mode.",)
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Disable ignoring EOS.",)
    parser.add_argument("--extra-request-body", metavar='{"key1": "value1", "key2": "value2"}', type=str, help="Append given JSON object to the request payload. You can use this to specify additional generate params like sampling params.",)
    args = parser.parse_args()

    run_benchmark(args)

