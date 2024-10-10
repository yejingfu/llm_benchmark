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
from async_request_sender import Context, AysncRequestSender, Metrics, calculate_metrics
import util


SYS_PROMPT="""
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
TMP_TEST_SHARED_PROMPT = False
PRINT_SAMPLES = 10

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")

    random.seed(1)
    np.random.seed(1)
    if not args.model:
        server_model = util.get_model(args.endpoint + "/models")
        if server_model is None and not args.model:
            raise RuntimeError("Failed to query model name from server")
        if not args.model:
            args.model = server_model
        assert args.model == server_model, f"Mismatched model name: {args.model}, {server_model}"
    logger.info(f"Model name: {args.model}")
    # get samples from dataset
    if args.sampling_policy == "nature":
        min_in_len = [args.min_prompt_len] * args.num_requests
        max_in_len = [args.max_prompt_len] * args.num_requests
        min_out_len = [args.min_output_len] * args.num_requests
        max_out_len = [args.max_output_len] * args.num_requests
    elif args.sampling_policy == "fixed":
        min_in_len = [args.fixed_prompt_len] * args.num_requests
        max_in_len = min_in_len
        min_out_len = [args.fixed_output_len] * args.num_requests
        max_out_len = min_out_len
    elif args.sampling_policy == "normal":
        min_in_len = np.rint(np.random.normal(args.prompt_len_mean, args.prompt_len_std, size=args.num_requests)).astype(np.int32)
        max_in_len = min_in_len
        min_out_len = np.rint(np.random.normal(args.output_len_mean, args.output_len_std, size=args.num_requests)).astype(np.int32)
        max_out_len = min_out_len
    if min_in_len is None:
        raise RuntimeError("Invalid input length and output length")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = util.load_requests_from_json(tokenizer, args.dataset, args.num_requests, min_in_len, max_in_len, min_out_len, max_out_len)
    logger.info(f"Got {len(samples)} requests")
    contexts = []
    for i in range(len(samples)):
        d = samples[i]
        if i < PRINT_SAMPLES and args.verbose:
            logger.info(f"Request[{i}]: {d[1]} / {d[2]}, {d[0][0: 100]}")
        contexts.append(Context(index=i, prompt=d[0], prompt_len=d[1], max_tokens=d[2]))

    # send requests async
    extra = {}
    ignore_eos = False if args.disable_ignore_eos else True
    sender = AysncRequestSender(args.endpoint, args.model, args.api_key, SYS_PROMPT if args.add_system_prompt else None, False if args.disable_stream else True, ignore_eos, args.verbose)
    logger.info("Warmup")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts[0:2], args.api_kind == "chat", 2, extra))
    end_time = time.perf_counter()
    logger.info(f"Warmup fininshed in {end_time - start_time} seconds")
    for i in range(2):
        contexts[i].clean()

    logger.info("Benchmark")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts, args.api_kind == "chat", args.parallel, extra))
    e2e_duration = time.perf_counter() - start_time
    logger.info(f"Benchmark fininshed in {e2e_duration} seconds")
    # metrics
    metrics = calculate_metrics(tokenizer, contexts, e2e_duration)
    if metrics is None:
        logger.warning("Failed to get metrics")
        return
    for ctx in contexts:
       if ctx.error:
            logger.warning(f"[{ctx.index}] ERROR: {ctx.error}")
       if not args.disable_warn_dismatch_output_len and abs(ctx.output_len - ctx.max_tokens) > 10:
            logger.warning(f"[{ctx.index}] Mismatched output length: expected {ctx.max_tokens}, got {ctx.output_len}")
    e2e_latency_p, ttft_p, tpot_p, tps_p = metrics.get_percentile([50, 90, 99])
    prompt_len, gen_len = 0, 0
    if args.sampling_policy == "fixed":
        prompt_len, gen_len = args.fixed_prompt_len, args.fixed_output_len
    elif args.sampling_policy == "normal":
        prompt_len, gen_len = args.prompt_len_mean, args.output_len_mean
    output = f"\n[BeginMetrics] {datetime.now().strftime('%m%d:%H-%M')}\n"
    output += f"log: {args.log_file}\n"
    output += f"model: {args.model}\n"
    output += f"sampling-policy: {args.sampling_policy}\n"
    output += f"sequence-length: {prompt_len}, {gen_len}\n"
    output += f"batch-size: {args.parallel}\n"
    output += f"e2e-latency(P50, P90, P99): {e2e_latency_p[0]:.2f}, {e2e_latency_p[1]:.2f}, {e2e_latency_p[2]:.2f}\n"
    output += f"ttft(P50, P90, P99): {ttft_p[0]:.2f}, {ttft_p[1]:.2f}, {ttft_p[2]:.2f}\n"
    output += f"tpot(P50, P90, P99): {tpot_p[0]:.3f}, {tpot_p[1]:.3f}, {tpot_p[2]:.3f}\n"
    output += f"tps(P50, P90, P99): {tps_p[0]:.1f}, {tps_p[1]:.1f}, {tps_p[2]:.1f}\n"
    output += f"throughput: {metrics.input_tokens/e2e_duration:.2f}, {metrics.output_tokens/e2e_duration:.2f}\n"
    output += f"rps: {len(contexts)/e2e_duration:.3f}\n"
    output += f"errors: {len(metrics.errors)}\n"
    if args.record_raw_metrics and args.log_file:
        output += f"raw-ttft: {metrics.ttft}\n"
        output += f"raw-tpot: {metrics.tpot}\n"
        output += f"raw-tps: {metrics.tps}\n"
    output += f"[EndMetrics]\n"
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
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--disable-stream", action="store_true", help="Disable stream mode")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Ignore EOS of the output")
    parser.add_argument("--disable-warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    # log
    parser.add_argument("--log-file", type=str, help="file to save log information")
    parser.add_argument("--record-raw-metrics", action="store_true", help="Dump raw metrics like TTFT or TPOT")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
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

