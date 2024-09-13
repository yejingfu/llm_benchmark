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
import resource
import time
from datetime import datetime

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional, Dict
from transformers import AutoTokenizer
from async_request_sender import Context, AysncRequestSender


SYS_PROMPT="""
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
TMP_TEST_SHARED_PROMPT = False
PRINT_SAMPLES = 10

def load_requests(tokenizer, args):
    logger.info(f"Load from dataset: {args.dataset}")
    dataset = []
    if os.path.exists(args.dataset):
        with open(args.dataset, "r") as f:
            data = json.load(f)
        if data is None:
            raise RuntimeError(f"Failed to load dataset {args.dataset}")
        if "kind" in data and data["kind"] == "ppio-internal":
            for d in data["data"]:
                dataset.append((d["prompt"], d["output"]))
        elif "sharegpt" in args.dataset.lower() or "share_gpt" in args.dataset.lower():
            dataset = [d for d in data if len(d["conversations"]) >= 2]
            dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset if len(data["conversations"][0]["value"]) > 10 and len(data["conversations"][1]["value"]) > 10]
    logger.info(f"The dataset has {len(dataset)} samples")
    if len(dataset) == 0:
        return None
    random.shuffle(dataset)
    pb = tqdm(total=args.num_requests, smoothing=0.0)
    output = []

    def _adjust_prompt(tokenizer, prompt, output, min_len, max_len, min_len_o, max_len_o):
        if min_len is None or min_len_o is None:
            return None, None, None
        max_len = min_len if max_len is None else max_len
        max_len_o = min_len_o if max_len_o is None else max_len_o
        prompt_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(output)
        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)
        if prompt_len < min_len:
            return None, None, None
        if prompt_len > max_len:
            prompt_len = random.randint(min_len, max_len)
            prompt = tokenizer.decode(prompt_tokens[0:prompt_len])
        if output_len > max_len_o or output_len < min_len_o:
            output_len = random.randint(min_len_o, max_len_o)
        return prompt, prompt_len, output_len

    if args.sampling_policy == "normal":
        norm_prompt_lens = np.rint(np.random.normal(args.prompt_len_mean, args.prompt_len_std, size=args.num_requests)).astype(np.int32)
        norm_output_lens = np.rint(np.random.normal(args.output_len_mean, args.output_len_std, size=args.num_requests)).astype(np.int32)

    for i in range(len(dataset)):
        if len(output) >= args.num_requests:
            break
        data = dataset[i]
        if args.sampling_policy == "nature":
            prompt, prompt_len, output_len = _adjust_prompt(tokenizer, data[0], data[1], args.min_prompt_len, args.max_prompt_len, args.min_output_len, args.max_output_len)
        elif args.sampling_policy == "fixed":
            prompt, prompt_len, output_len = _adjust_prompt(tokenizer, data[0], data[1], args.fixed_prompt_len, args.fixed_prompt_len, args.fixed_output_len, args.fixed_output_len)
        elif args.sampling_policy == "normal":
            n = len(output)
            prompt, prompt_len, output_len = _adjust_prompt(tokenizer, data[0], data[1], norm_prompt_lens[n], norm_prompt_lens[n], norm_output_lens[n], norm_output_lens[n])
        else:
            raise ValueError(f"Unknown sampling policy: {args.samping_policy}")
        if prompt is not None:
            output.append((prompt, prompt_len, output_len))
            pb.update(1)
    pb.close()
    return output

def get_model(url: str, headers = None)->Optional[str]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    return model_list[0]["id"] if model_list else None

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")

    random.seed(1)
    np.random.seed(1)
    if not args.model:
        server_model = get_model(args.endpoint + "/models")
        if server_model is None and not args.model:
            raise RuntimeError("Failed to query model name from server")
        if not args.model:
            args.model = server_model
        assert args.model == server_model, f"Mismatched model name: {args.model}, {server_model}"
    logger.info(f"Model name: {args.model}")
    # get samples from dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = load_requests(tokenizer, args)
    logger.info(f"Got {len(samples)} requests")
    contexts = []
    for i in range(len(samples)):
        d = samples[i]
        if i < PRINT_SAMPLES and args.print_verbose:
            logger.info(f"Request[{i}]: {d[1]} / {d[2]}, {d[0][0: 100]}")
        contexts.append(Context(index=i, prompt=d[0], prompt_len=d[1], max_tokens=d[2]))

    # send requests async
    extra = {}
    sender = AysncRequestSender(args.endpoint, args.model, args.api_key, SYS_PROMPT if args.add_system_prompt else None, False if args.disable_stream else True, args.ignore_eos, args.print_verbose)
    logger.info("Warmup")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts[0:2], args.api_kind == "chat", 2, extra))
    end_time = time.perf_counter()
    logger.info(f"Warmup fininshed in {end_time - start_time} seconds")
    logger.info("Benchmark")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts, args.api_kind == "chat", args.parallel, extra))
    e2e_duration = time.perf_counter() - start_time
    logger.info(f"Benchmark fininshed in {e2e_duration} seconds")
    # metrics
    num_errors = 0
    e2e_latency = []
    ttft = []
    tpot = []
    input_tokens = 0
    output_tokens = 0
    for i in range(len(contexts)):
        ctx = contexts[i]
        if ctx.error:
            num_errors += 1
            logger.warning(f"[{ctx.index}] ERROR: {ctx.error}")
        else:
            ctx.output_len = len(tokenizer.encode(ctx.generated))
            if ctx.output_len == 0:
                logger.warning(f"[{ctx.index}] Empty output")
                continue
            if ctx.ttft is None:
                logger.warning(f"[{ctx.index}] Invalid ttft")
            else:
                ctx.decode_latency = ctx.e2e_latency - ctx.ttft
                ctx.tpot = ctx.decode_latency / ctx.output_len
                input_tokens += ctx.prompt_len
                output_tokens += ctx.output_len
                e2e_latency.append(ctx.e2e_latency)
                ttft.append(ctx.ttft)
                tpot.append(ctx.tpot)
                if args.warn_dismatch_output_len and abs(ctx.output_len - ctx.max_tokens) > 10:
                    logger.info(f"[{ctx.index}] Mismatched output length: expected {ctx.max_tokens}, got {ctx.output_len}")
    PERCENTILES = [50, 90, 99]
    e2e_latency_p = np.percentile(e2e_latency, PERCENTILES)
    ttft_p = np.percentile(ttft, PERCENTILES)
    tpot_p = np.percentile(tpot, PERCENTILES)
    output = f"\n===== Metrics @ {datetime.now().strftime('%m%d:%H-%M')} , duration: {e2e_duration} =====\n"
    output += f"model: {args.model}, policy: {args.sampling_policy}, prompt-len:{contexts[i].prompt_len}, output-len:{contexts[i].max_tokens}, parallel: {args.parallel}\n"
    output += f"e2e latency(Median, P90, P99): {e2e_latency_p[0]:.2f}, {e2e_latency_p[1]:.2f}, {e2e_latency_p[2]:.2f}\n"
    output += f"ttft(Median, P90, P99): {ttft_p[0]:.2f}, {ttft_p[1]:.2f}, {ttft_p[2]:.2f}\n"
    output += f"tpot(Median, P90, P99): {tpot_p[0]:.3f}, {tpot_p[1]:.3f}, {tpot_p[2]:.3f}\n"
    output += f"throughput(input, output): {input_tokens/e2e_duration:.2f}, {output_tokens/e2e_duration:.2f}\n"
    print(output)
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # LLM Server
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--backend", type=str, default="vllm", help="The backend e.g. vllm, trtllm, default is vllm")
    parser.add_argument("--api-key", type=str, help="The api key to call commercial inference API")
    parser.add_argument("--api-kind", type=str, default="completions", choices=["chat", "completions"], help="Can be: chat or completions(default)")
    # input parameters
    parser.add_argument("--model", type=str, help="The model name, if not set, call 'endpoint/models' to query")
    # test data sampling
    parser.add_argument("--sampling-policy", type=str, default="nature", choices=["nature", "fixed", "normal"])
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--fixed-prompt-len", type=int, default=3500)
    parser.add_argument("--fixed-output-len", type=int, default=500)
    parser.add_argument("--prompt-len-mean", type=int, default=550)
    parser.add_argument("--prompt-len-std", type=int, default=150)
    parser.add_argument("--output-len-mean", type=int, default=150)
    parser.add_argument("--output-len-std", type=int, default=20)
    # press test setting
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of prompts for benckmark.")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--dataset", type=str, help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--disable-stream", action="store_true", help="Disable stream mode")
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS of the output")
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    parser.add_argument("--log-file", type=str, help="file to save log information")
    parser.add_argument("--print-verbose", action="store_true", help="print in verbose mode")
    args = parser.parse_args()

    # set_ulimit: target_soft_limit=65535
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < 65535:
        try:
            resource.setrlimit(resource_type, (65535, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")
    main(args)

