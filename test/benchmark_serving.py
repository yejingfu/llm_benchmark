import aiohttp
import argparse
import asyncio
import json
import numpy as np
import os
import random
import re
import requests
import subprocess
import sys
import time

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

#system_prompt = ""
# 740 Tokens
system_prompt = """
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
#system_prompt = system_prompt * 2

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

PERCENTILES = [25, 50, 75, 90, 95, 99, 99.9, 99.99]

REQUIRED_KEYS_BY_POLICY = {
    "nature": {
        "max_turns",
        "min_prompt_len",
        "min_output_len",
        "max_prompt_len",
        "max_prompt_output_len",
    },
    "fixed": {"fixed_prompt_len", "fixed_output_len"},
    "normal": {
        "max_seq_len",
        "prompt_len_mean",
        "prompt_len_std",
        "output_len_mean",
        "output_len_std",
    },
}


@dataclass
class Response:
    prompt: str = field(default="")
    generated: str = field(default="")
    prompt_len: int = field(default=0)
    output_len: int = field(default=0)
    latency: float = field(default=0.0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)


REQUEST_RESPONSES: List[Response] = []


async def update_sem(
    sem: asyncio.Semaphore,
    update_interval: int,
    ramp_up_period: int,
    max_concurrent_requests: int,
):
    n_parts = ramp_up_period // update_interval
    base = max_concurrent_requests // n_parts
    remainder = max_concurrent_requests % n_parts
    partitions = [base] * (n_parts - remainder) + [base + 1] * remainder
    for p in partitions:
        await asyncio.sleep(update_interval)
        for _ in range(p):
            sem.release()


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code=trust_remote_code, **kwargs
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return tokenizer


def wait_for_server_ready(backend, port, wait_time_secs=900):
    if backend == "siliconllm":
        url = f"http://localhost:{port}/v1/ready"
    elif backend == "vllm":
        url = f"http://localhost:{port}/health"
    elif backend == "trtllm":
        url = f"http://localhost:{port}/v2/health/ready"
    elif backend == "tgi":
        url = f"http://localhost:{port}/health"
    logger.info(f"Waiting for server ready at {url} ...")

    check_container_interval = 10
    wait_secs = wait_time_secs
    container_name = f"benchmark_{backend}_server"

    while wait_secs > 0:
        time.sleep(5)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info(f"The server is ready: {url}")
                return
        except requests.RequestException as e:
            #logger.warning(f"The server return bad response: {url}, {e}")
            pass

        if (wait_time_secs - wait_secs) % check_container_interval == 0:
            containers = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                stdout=subprocess.PIPE,
                text=True,
            )
            if container_name in containers.stdout:
                container_id = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--format",
                        "{{.ID}}",
                        "--filter",
                        f"name={container_name}",
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.strip()
                container_status = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "-f",
                        f"id={container_id}",
                        "--format",
                        "{{.Status}}",
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.strip()

                if "Exited" in container_status:
                    print(
                        "=== Benchmark container has exited abnormal. Outputting container log ==="
                    )
                    subprocess.run(["docker", "logs", container_name])
                    print("=== Benchmark container log end ===")
                    exit(1)
                else:
                    print(f"waiting for launch {backend} server ... if awlays failed, please make sure http_proxy is unset")

        wait_secs -= 1

    print(f"=== Timeout {wait_time_secs} secs. Server not ready.")
    exit(1)


def prepare_dataset(dataset_path: str) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"{len(dataset)} samples loaded")

    cleaned_dataset = []
    for sample in tqdm(dataset):
        cleaned_conversations = []
        for msg in sample["conversations"]:
            if len(msg["value"]) == 0:
                continue
            if len(cleaned_conversations) % 2 == 0:
                if msg["from"] in ["user", "human"]:
                    cleaned_conversations.append(msg["value"])
                else:
                    continue
            else:
                if msg["from"] in ["gpt", "chatgpt", "bing", "bard"]:
                    cleaned_conversations.append(msg["value"])
                else:
                    continue
        if len(cleaned_conversations) % 2 == 1:
            cleaned_conversations = cleaned_conversations[:-1]
        if len(cleaned_conversations) != 0:
            cleaned_dataset.append(cleaned_conversations)
    logger.info(f"cleanup done, {len(cleaned_dataset)} samples after cleanup")
    return cleaned_dataset


def validate_policy_params(policy, kwargs):
    required_keys = REQUIRED_KEYS_BY_POLICY[policy]
    if not required_keys.issubset(kwargs.keys()) or not set(kwargs.keys()).issubset(
        required_keys
    ):
        raise ValueError(f"Missing or extra parameters for '{policy}' policy")


def sample_nature_policy(
    dataset, num_requests, tokenizer, add_system_prompt, **kwargs
) -> List[Tuple[str, int, int]]:
    sampled_dataset = []
    max_turns = kwargs["max_turns"]
    min_prompt_len = kwargs["min_prompt_len"]
    min_output_len = kwargs["min_output_len"]
    max_prompt_len = kwargs["max_prompt_len"]
    max_prompt_output_len = kwargs["max_prompt_output_len"]
    system_msg = ""
    if add_system_prompt:
        system_msg = f"<<SYS>>\n{system_prompt}<</SYS>>\n\n"
    pb = tqdm(total=num_requests, smoothing=0.0)
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = [dataset[i] for i in permutation]
    for sample in shuffled_dataset:
        turns = min(len(sample) // 2, max_turns)
        selected_turn = np.random.randint(0, turns)
        prompt = ""
        prompt_output = ""
        has_ascii = False
        for idx, msg in enumerate(sample[: (selected_turn + 1) * 2]):
            if any(ord(c) >= 128 for c in msg):
                has_ascii = True
                break
            if idx % 2 == 0:
                if idx == 0:
                    prompt += f"<s>[INST] {system_msg}{msg} [/INST]"
                else:
                    prompt += f"<s>[INST] {msg} [/INST]"
            else:
                if idx == selected_turn * 2 + 1:
                    prompt_output = prompt + f"{msg}</s>"
                else:
                    prompt += f"{msg}</s>"

        if has_ascii:  # skip msg with non-ascii characters.
            continue

        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > max_prompt_len or len(prompt_tokens) < min_prompt_len:
            continue

        prompt_output_tokens = tokenizer.encode(prompt_output)
        if (
            len(prompt_output_tokens) > max_prompt_output_len
            or len(prompt_output_tokens) - len(prompt_tokens) < min_output_len
        ):
            continue

        sampled_dataset.append(
            (prompt, len(prompt_tokens), len(prompt_output_tokens) - len(prompt_tokens))
        )
        pb.update(1)
        if len(sampled_dataset) == num_requests:
            break
    pb.close()
    return sampled_dataset


def sample_fixed_policy(
    dataset, num_requests, tokenizer, add_system_prompt, **kwargs
) -> List[Tuple[str, int, int]]:
    sampled_dataset = []
    fixed_prompt_len = kwargs["fixed_prompt_len"]
    fixed_output_len = kwargs["fixed_output_len"]
    system_msg = ""
    if add_system_prompt:
        system_msg = f"<<SYS>>\n{system_prompt}<</SYS>>\n\n"
        prompt_tokens = tokenizer.encode(system_msg)
        print(f"======system_prompt len: {len(prompt_tokens)}")

    pb = tqdm(total=num_requests, smoothing=0.0)
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = [dataset[i] for i in permutation]
    for sample in shuffled_dataset:
        prompt = ""
        for idx, msg in enumerate(sample):
            if idx % 2 == 0:
                if idx == 0:
                    prompt += f"<s>[INST] {system_msg}{msg} [/INST]"
                else:
                    prompt += f"<s>[INST] {msg} [/INST]"
            else:
                prompt += f"{msg}</s>"

        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) < fixed_prompt_len:
            continue
        prompt_tokens = prompt_tokens[:fixed_prompt_len]
        prompt = tokenizer.decode(prompt_tokens)
        sampled_dataset.append((prompt, fixed_prompt_len, fixed_output_len))
        pb.update(1)
        if len(sampled_dataset) == num_requests:
            break
    pb.close()
    return sampled_dataset


def sample_normal_policy(
    dataset, num_requests, tokenizer, add_system_prompt, **kwargs
) -> List[Tuple[str, int, int]]:
    sampled_dataset = []
    max_seq_len = kwargs["max_seq_len"]
    prompt_len_mean = kwargs["prompt_len_mean"]
    prompt_len_std = kwargs["prompt_len_std"]
    output_len_mean = kwargs["output_len_mean"]
    output_len_std = kwargs["output_len_std"]
    system_msg = ""
    if add_system_prompt:
        system_msg = f"<<SYS>>\n{system_prompt}<</SYS>>\n\n"
    pb = tqdm(total=num_requests, smoothing=0.0)
    weights = [
        sum([len(msg) for msg in sample]) * len(dataset) + idx
        for idx, sample in enumerate(dataset)
    ]
    sorted_indices = np.argsort(weights)
    sorted_dataset = [dataset[i] for i in sorted_indices]
    prompt_lens = np.rint(
        np.random.normal(prompt_len_mean, prompt_len_std, size=num_requests)
    ).astype(np.int64)
    output_lens = np.rint(
        np.random.normal(output_len_mean, output_len_std, size=num_requests)
    ).astype(np.int64)
    sorted_indices = np.argsort(prompt_lens + output_lens)
    prompt_lens = prompt_lens[sorted_indices]
    output_lens = output_lens[sorted_indices]
    for sample in sorted_dataset:
        prompt_len = prompt_lens[len(sampled_dataset)]
        output_len = output_lens[len(sampled_dataset)]
        if prompt_len <= 0:
            prompt_len = 1
        if prompt_len > max_seq_len:
            prompt_len = max_seq_len - 1
        if output_len <= 0:
            output_len = 1
        if output_len + prompt_len > max_seq_len:
            output_len = max_seq_len - prompt_len
        prompt = ""
        for idx, msg in enumerate(sample):
            if idx % 2 == 0:
                if idx == 0:
                    prompt += f"<s>[INST] {system_msg}{msg} [/INST]"
                else:
                    prompt += f"<s>[INST] {msg} [/INST]"
            else:
                prompt += f"{msg}</s>"

        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) < prompt_len:
            continue
        prompt_tokens = prompt_tokens[:prompt_len]
        prompt = tokenizer.decode(prompt_tokens)
        sampled_dataset.append((prompt, int(prompt_len), int(output_len)))
        pb.update(1)
        if len(sampled_dataset) == num_requests:
            permutation = np.random.permutation(len(sampled_dataset))
            sampled_dataset = [sampled_dataset[i] for i in permutation]
            break
    pb.close()
    return sampled_dataset


def sample_requests(
    dataset,
    num_warmup_requests,
    num_benckmark_requests,
    tokenizer,
    add_system_prompt,
    policy,
    **kwargs,
):
    num_requests = num_warmup_requests + num_benckmark_requests
    validate_policy_params(policy, kwargs)
    if policy == "nature":
        sampled_requests = sample_nature_policy(
            dataset, num_requests, tokenizer, add_system_prompt, **kwargs
        )
    elif policy == "fixed":
        sampled_requests = sample_fixed_policy(
            dataset, num_requests, tokenizer, add_system_prompt, **kwargs
        )
    elif policy == "normal":
        sampled_requests = sample_normal_policy(
            dataset, num_requests, tokenizer, add_system_prompt, **kwargs
        )
    else:
        raise ValueError(f"Invalid policy: {policy}")

    if args.pad_requests:
        while len(sampled_requests) < num_requests:
            sampled_requests += sampled_requests[: num_requests - len(sampled_requests)]
    else:
        assert (
            len(sampled_requests) == num_requests
        ), f"expected {num_requests} requests, but sampled {len(sampled_requests)}"

    logger.info(
        f"prompt len: mean={np.mean([item[1] for item in sampled_requests]):.2f}, std={np.std([item[1] for item in sampled_requests]):.2f}"
    )
    logger.info(
        f"output len: mean={np.mean([item[2] for item in sampled_requests]):.2f}, std={np.std([item[2] for item in sampled_requests]):.2f}"
    )

    sampled_requests_for_warmup = sampled_requests[:num_warmup_requests]
    sampled_requests_for_benchmark = sampled_requests[num_warmup_requests:]
    logger.info(
        f"{len(sampled_requests_for_warmup)} requests for warmup, {len(sampled_requests_for_benchmark)} requests for benckmark"
    )
    total_tokens = lambda requests: sum(
        prompt_len + output_len for _, prompt_len, output_len in requests
    )
    avg_prompt_len = lambda requests: np.mean([p for _, p, _ in requests])
    avg_generate_len = lambda requests: np.mean([g for _, _, g in requests])
    logger.info(
        f"{total_tokens(sampled_requests_for_warmup)} tokens for warmup, {total_tokens(sampled_requests_for_benchmark)} tokens for benchmark"
    )
    logger.info(
        f"avg {avg_prompt_len(sampled_requests_for_warmup)} tokens for warmup, {avg_prompt_len(sampled_requests_for_benchmark)} tokens for benchmark"
    )
    logger.info(
        f"avg {avg_generate_len(sampled_requests_for_warmup)} generate tokens for warmup, {avg_generate_len(sampled_requests_for_benchmark)} generate tokens for benchmark"
    )
    return sampled_requests_for_warmup, sampled_requests_for_benchmark

async def send_request(
    semaphore: asyncio.Semaphore,
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    n: int,
    best_of: int,
    use_beam_search: bool,
    do_sample: bool,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
    top_k: int,
    stream: bool,
    model: str,
    eos_token_id: int,
) -> None:
    async with semaphore:
        request_start_time = time.perf_counter()

        headers = {"User-Agent": "Benchmark Client"}
        if backend == "siliconllm":
            pload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if do_sample else -1,
                "max_tokens": output_len,
                "logit_bias": {eos_token_id: -100.0},
                "stream": stream,
            }
        elif backend == "vllm":
            pload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if do_sample else -1,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": stream,
            }
        elif backend == "trtllm":
            assert not use_beam_search
            pload = {
                "text_input": prompt,
                "max_tokens": output_len,
                "bad_words": "",
                "stop_words": "",
                "stream": stream,
            }
            if do_sample:
                pload["temperature"] = temperature
                pload["top_p"] = top_p
                pload["top_k"] = top_k
        elif backend == "tgi":
            assert not use_beam_search
            params = {
                "max_new_tokens": output_len,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "seed": 2023,
            }
            if do_sample:
                params["temperature"] = temperature
                params["top_p"] = top_p
                params["top_k"] = top_k
            pload = {
                "inputs": prompt,
                "parameters": params,
                "stream": stream,
            }
        elif backend == "lightllm":
            assert not use_beam_search
            pload = {
                "inputs": prompt,
                "parameters": {
                    "do_sample": do_sample,
                    "ignore_eos": True,
                    "max_new_tokens": output_len,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        timeout = aiohttp.ClientTimeout(total=3600 * 3)
        MAX_RETRIES = 10
        first_token_time = None
        ttft = None
        tpot = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.post(
                        api_url, headers=headers, json=pload
                    ) as response:
                        chunks = []
                        if backend == "tgi":
                            async for chunk in response.content:
                                if chunk == b"\n":
                                    continue
                                #logger.info(f"tgi chunk: {chunk}")
                                if ttft is None:
                                    first_token_time = time.perf_counter()
                                    ttft = first_token_time - request_start_time
                                chunks.append(chunk)
                        else:
                            async for chunk, _ in response.content.iter_chunks():
                                if ttft is None:
                                    first_token_time = time.perf_counter()
                                    ttft = first_token_time - request_start_time
                                chunks.append(chunk)

                    if output_len > 1:
                        tpot = (time.perf_counter() - first_token_time) / (
                            output_len - 1
                        )
                    if stream:
                        outputs = []
                        for chunk in chunks:
                            try:
                                data = chunk.rstrip(b"\x00").lstrip(b"data:").rstrip(b"\n\n").strip().decode("utf-8")
                                output = json.loads(data)
                                if backend == "trtllm":
                                    text = output["text_output"]
                                elif backend == "tgi":
                                    #text = output["generated_text"]
                                    text = output["token"]["text"]
                                else:
                                    text = output["choices"][0]["text"]
                                #logger.info(f"chunk: {len(text)}/{output_len}")
                                outputs.append(text)
                            except json.decoder.JSONDecodeError as err:
                                # logger.warning(f"{err}, data: {data}")
                                continue
                            except Exception as err:
                                # exception for tgi output: b":\n\n""
                                logger.warning(f"err:{err}, raw chunk: {chunk}")
                                continue
                        output = "".join(outputs)
                        #logger.info(f"stream output: {output}")
                    else:
                        output = b"".join(chunks).decode("utf-8")
                        output = json.loads(output)
                    break
                except aiohttp.ClientError as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying")
                    else:
                        raise Exception(f"All {MAX_RETRIES} attempts failed: {e}")

        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        global REQUEST_RESPONSES
        REQUEST_RESPONSES.append(
            Response(
                prompt=prompt,
                generated=output,
                prompt_len=prompt_len,
                output_len=output_len,
                latency=request_latency,
                ttft=ttft,
                tpot=tpot,
            )
        )


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    max_concurrent_requests: int,
    ramp_up_period: int,
    n: int,
    best_of: int,
    use_beam_search: bool,
    do_sample: bool,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
    top_k: int,
    stream: bool,
    model: str,
    eos_token_id: int,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    for _ in range(max_concurrent_requests - 1):
        await semaphore.acquire()
    tasks: List[asyncio.Task] = [
        asyncio.create_task(
            update_sem(semaphore, 1, ramp_up_period, max_concurrent_requests - 1)
        )
    ]
    progress_bar = async_tqdm(
        total=len(input_requests), desc="Processing Requests", smoothing=0.0
    )
    for request in input_requests:
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                semaphore,
                backend,
                api_url,
                prompt,
                prompt_len,
                output_len,
                n,
                best_of,
                use_beam_search,
                do_sample,
                presence_penalty,
                frequency_penalty,
                repetition_penalty,
                temperature,
                top_p,
                top_k,
                stream,
                model,
                eos_token_id,
            )
        )
        tasks.append(task)
        task.add_done_callback(lambda _: progress_bar.update())
    await asyncio.gather(*tasks)
    progress_bar.close()


def main(args: argparse.Namespace):
    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/{args.endpoint}"
    tokenizer = get_tokenizer(
        args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=args.use_fast
    )
    cleaned_dataset = prepare_dataset(args.dataset)
    if args.sampling_policy == "nature":
        requests_for_warmup, requests_for_benchmark = sample_requests(
            cleaned_dataset,
            args.num_warmup_requests,
            args.num_benchmark_requests,
            tokenizer,
            args.add_system_prompt,
            "nature",
            max_turns=args.max_turns,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            min_output_len=args.min_output_len,
            max_prompt_output_len=args.max_prompt_output_len,
        )
    elif args.sampling_policy == "fixed":
        requests_for_warmup, requests_for_benchmark = sample_requests(
            cleaned_dataset,
            args.num_warmup_requests,
            args.num_benchmark_requests,
            tokenizer,
            args.add_system_prompt,
            "fixed",
            fixed_prompt_len=args.fixed_prompt_len,
            fixed_output_len=args.fixed_output_len,
        )
    elif args.sampling_policy == "normal":
        requests_for_warmup, requests_for_benchmark = sample_requests(
            cleaned_dataset,
            args.num_warmup_requests,
            args.num_benchmark_requests,
            tokenizer,
            args.add_system_prompt,
            "normal",
            max_seq_len=args.max_seq_len,
            prompt_len_mean=args.prompt_len_mean,
            prompt_len_std=args.prompt_len_std,
            output_len_mean=args.output_len_mean,
            output_len_std=args.output_len_std,
        )
    else:
        raise ValueError(f"Invalid policy: {args.sampling_policy}")

    wait_for_server_ready(args.backend, args.port, wait_time_secs=900)

    global REQUEST_RESPONSES
    for phase, input_requests in zip(
        ("Warmup", "Benchmark"), (requests_for_warmup, requests_for_benchmark)
    ):
        start_time = time.perf_counter()
        asyncio.run(
            benchmark(
                args.backend,
                api_url,
                input_requests,
                args.max_concurrent_requests,
                args.ramp_up_period,
                args.n,
                args.best_of,
                args.use_beam_search,
                args.do_sample,
                args.presence_penalty,
                args.frequency_penalty,
                args.repetition_penalty,
                args.temperature,
                args.top_p,
                args.top_k if args.top_k > 0 else tokenizer.vocab_size,
                args.stream,
                args.model,
                tokenizer.eos_token_id,
            )
        )
        end_time = time.perf_counter()
        if phase == "Benchmark":
            REQUEST_RESPONSES = REQUEST_RESPONSES[args.num_warmup_requests :]
        else:
            continue  # skip warmup phase

        generate_tokens = 0
        for response in REQUEST_RESPONSES:
            generated_tokens = tokenizer.encode(response.generated, add_special_tokens=False)
            generated_len = len(generated_tokens)
            generate_tokens += generated_len
            if abs(generated_len - response.output_len) > 20 and args.warn_dismatch_output_len:
                logger.warning(
                    f"expect generated {response.output_len} tokens"
                    f", but got {len(generated_tokens)}."
                    #f" generated text: {repr(response.generated)}"
                    #f", generated tokens: {generated_tokens}"
                )
            if args.generated_log_file:
                record = asdict(response)
                record["generated_len"] = generated_len
                with open(args.generated_log_file, "a") as f:
                    json.dump(record, f)
                    f.write("\n")

        result_data = OrderedDict(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "backend": args.backend,
                "num_warmup_requests": args.num_warmup_requests,
                "num_benchmark_requests": args.num_benchmark_requests,
                "max_concurrent_requests": args.max_concurrent_requests,
                "sampling_policy": args.sampling_policy,
            }
        )
        duration = end_time - start_time
        logger.info(f"{phase} Total time: {duration:.2f} s")
        result_data["total_time"] = round(duration, 2)
        total_tokens = np.sum(
            [
                response.prompt_len for response in REQUEST_RESPONSES
            ]
        )
        total_tokens += generate_tokens
        ## calculate price
        GPU_RMB_PER_HOUR = 2.69
        GROSS_MARGIN = 0.3
        GPU_USAGE = 0.5
        cost_per_gpu_sec = 2.69 / 7.0 / 3600
        tokens_per_sec_per_gpu = total_tokens / duration / args.gpus
        price_per_mtokens = cost_per_gpu_sec * 1000000 / GPU_USAGE / tokens_per_sec_per_gpu # / (1 - GROSS_MARGIN)
        logger.info(
            f"{phase} Throughput: {len(input_requests) / duration:.2f} ({len(input_requests) / duration / 8.0:.2f}) requests/s, {total_tokens / duration:.2f} ({generate_tokens / duration:.2f}) tokens/s, {total_tokens / args.gpus / duration:.2f} ({generate_tokens / args.gpus / duration:.2f}) tokens/s/gpu, price: {price_per_mtokens:0.3f}, {price_per_mtokens / (1 - GROSS_MARGIN):0.3f}"
        )
        result_data["tokens_per_second"] = int(total_tokens / duration)
        result_data["output_tokens_per_second"] = int(generate_tokens / duration)
        result_data["requests_per_second"] = round(len(input_requests) / duration, 2)

        # Compute the latency statistics.
        latencies = [response.latency for response in REQUEST_RESPONSES]
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        logger.info(f"{phase} Average latency: {avg_latency:.2f} s")
        logger.info(f"{phase} Minimum latency: {min_latency:.2f} s")
        logger.info(f"{phase} Maximum latency: {max_latency:.2f} s")
        result_data["avg latency"] = round(avg_latency, 3)
        result_data["min_latency"] = round(min_latency, 3)
        result_data["max_latency"] = round(max_latency, 3)
        latency_percentile = ", ".join(
            [
                f"P{k} = {v:.3f}"
                for k, v in zip(PERCENTILES, np.percentile(latencies, PERCENTILES))
            ]
        )
        logger.info(
            f"{phase} Latency: avg = {np.mean(latencies):.3f}, {latency_percentile}"
        )
        result_data["latency@Avg"] = round(np.mean(latencies), 3)
        for percentile, v in zip(PERCENTILES, np.percentile(latencies, PERCENTILES)):
            result_data[f"latency@P{percentile}"] = round(v, 3)

        avg_per_token_latency = np.mean(
            [
                response.latency / (response.prompt_len + response.output_len)
                for response in REQUEST_RESPONSES
            ]
        )
        logger.info(f"{phase} Average latency per token: {avg_per_token_latency:.2f} s")
        result_data["avg latency per prompt token"] = round(avg_per_token_latency, 3)
        avg_per_output_token_latency = np.mean(
            [response.latency / response.output_len for response in REQUEST_RESPONSES]
        )
        logger.info(
            f"{phase} Average latency per output token: "
            f"{avg_per_output_token_latency:.2f} s"
        )
        result_data["avg latency per output token"] = round(
            avg_per_output_token_latency, 3
        )

        if args.stream:
            ttft = [response.ttft for response in REQUEST_RESPONSES]
            ttft_percentile = ", ".join(
                [
                    f"P{k} = {v:.3f}"
                    for k, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES))
                ]
            )
            logger.info(f"{phase} TTFT: avg = {np.mean(ttft):.3f}({(total_tokens - generate_tokens) / len(input_requests) / np.mean(ttft):.3f}), {ttft_percentile}")
            result_data["TTFT@Avg"] = round(np.mean(ttft), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES)):
                result_data[f"TTFT@P{percentile}"] = round(v, 3)
            tpot = [response.tpot or np.nan for response in REQUEST_RESPONSES]
            tpot_percentile = ", ".join(
                [
                    f"P{k} = {v:.3f}"
                    for k, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES))
                ]
            )
            logger.info(f"{phase} TPOT: avg = {np.mean(tpot):.3f} ({1 / np.mean(tpot):.3f}), {tpot_percentile}")
            result_data["TPOT@Avg"] = round(np.mean(tpot), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES)):
                result_data[f"TPOT@P{percentile}"] = round(v, 3)

        new_data = OrderedDict()
        for k, v in result_data.items():
            k = k.lower()
            new_data[k] = v
        for k, v in result_data.items():
            if k not in new_data:
                new_data[k] = v

        with open(args.log_file, "a") as f:
            json.dump(new_data, f)
            f.write("\n")


def simple_verify_args(args):
    # As not all frameworks support these generation params like use_beam_search, presence_penalty, best_of etc.
    # and the most commonly generation strategy is greedy search & random sample with params `top_p, top_k, temperature`
    # so we just simply verify the args here.
    assert not args.use_beam_search, "do not support benchmark beam search now."
    assert (
        args.best_of == 1 and args.n == 1
    ), "do not support benchmark best_of and n now."
    assert (
        args.presence_penalty == 0.0
        and args.frequency_penalty == 0.0
        and args.repetition_penalty == 1.0
    ), "do not support benchmark penalty policies now."
    if args.do_sample:
        assert (
            args.temperature > 0.0
        ), "temperature must be greater than 0.0 when do_sample is True."
    else:
        assert args.top_k == 1 and args.top_p == 1.0 and args.temperature == 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["trtllm", "vllm", "tgi", "siliconllm"],
    )
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint", type=str, default="v1/completions"
    )  # `generate_stream` for TGI/trtllm
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="How many sequences to generate for each prompt.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and returns the "
        "top `n` results, with the default value of `n` being one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--sampling-policy",
        type=str,
        default="nature",
        choices=["nature", "fixed", "normal"],
    )
    parser.add_argument("--max_turns", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--generate-length", type=int, default=64)
    parser.add_argument("--min_prompt_len", type=int, default=4)
    parser.add_argument("--min_output_len", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--max_prompt_output_len", type=int, default=4096)
    parser.add_argument("--fixed_prompt_len", type=int, default=128)
    parser.add_argument("--fixed_output_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--prompt_len_mean", type=int, default=550)
    parser.add_argument("--prompt_len_std", type=int, default=150)
    parser.add_argument("--output_len_mean", type=int, default=150)
    parser.add_argument("--output_len_std", type=int, default=20)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter")
    parser.add_argument("--top_k", type=int, default=1, help="top-k parameter")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature parameter"
    )
    parser.add_argument("--presence_penalty", type=float, default=0.0)  # TODO
    parser.add_argument("--frequency_penalty", type=float, default=0.0)  # TODO
    parser.add_argument("--repetition_penalty", type=float, default=1.0)  # TODO
    parser.add_argument(
        "--decode-strategy", type=str, required=False, choices=[]
    )  # TODO
    parser.add_argument("--quentize", type=str, required=False, choices=[])  # TODO
    parser.add_argument(
        "--num-warmup-requests",
        type=int,
        default=1000,
        help="Number of prompts for warmup.",
    )
    parser.add_argument(
        "--num-benchmark-requests",
        type=int,
        default=8000,
        help="Number of prompts for benckmark.",
    )
    parser.add_argument("--max-concurrent-requests", type=int, default=400)
    parser.add_argument("--ramp_up_period", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument(
        "--add-system-prompt",
        action="store_true",
        help="add system prompt in front of each conversation",
    )
    parser.add_argument(
        "--pad-requests",
        action="store_true",
        help="pad the requests repeatedly when the number of sampled requests is less than expected",
    )
    parser.add_argument(
        "--warn-dismatch-output-len",
        action="store_true",
        help="warn when generated tokens number is not equal to expected output_len",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="file to save log information",
    )
    parser.add_argument(
        "--generated-log-file",
        type=str,
        default=None,
        help="file to save generated text",
    )
    parser.add_argument("--gpus", type=int, default=8, help="The total num of GPUs")
    args = parser.parse_args()
    args.prompt_len_mean = args.context_length
    args.output_len_mean = args.generate_length
    # args.fixed_prompt_len = args.context_length
    # args.fixed_output_len = args.generate_length

    simple_verify_args(args)

    main(args)
