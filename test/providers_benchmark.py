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

PRINT_PROVIDERS = 10
PRINT_SAMPLES = 10
## valid providers: Lepton, OctoAI, Novita, Together, DeepInfra, Replicate, Fireworks, Groq, DeepSeek, OpenAI, 01.AI
OPENROUTER_EP = "https://openrouter.ai/api/v1"

def run_benchmark(provider: util.LlmProvider, contexts: List[Context], tokenizer: AutoTokenizer, args: argparse.Namespace):
    logger.info(f"Test provider: {provider}")
    for ctx in contexts:
        ctx.clean()
    extra = {}
    if provider.is_openrouter():
        extra["provider"] = {
            "order": [provider.get_openrouter_provider()],
            "allow_fallbacks": False
        }
    sender = AysncRequestSender(provider.endpoint, provider.model, provider.api_key, None, True, True, args.verbose)
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts, args.api_kind == "chat", args.parallel, extra))
    e2e_duration = time.perf_counter() - start_time
    metrics = calculate_metrics(tokenizer, contexts, e2e_duration)
    if metrics is None:
        logger.warning(f"Failed to get metrics from provider: {provider}")
        return
    for i in range(len(metrics.errors)):
        logger.warning(f"Error[{i}]: {metrics.errors[i]}")
    e2e_latency_p, ttft_p, tpot_p, tps_p = metrics.get_percentile([50, 90, 99])
    output = f"\n===== Metrics @ {datetime.now().strftime('%m%d:%H-%M')} , duration: {e2e_duration} =====\n"
    output += f"privder: {provider} \n"
    output += f"prompt-len: {args.input_len}, output-len: {args.output_len}, parallel: {args.parallel} \n"
    output += f"e2e latency(Median, P90, P99): {e2e_latency_p[0]:.2f}, {e2e_latency_p[1]:.2f}, {e2e_latency_p[2]:.2f}\n"
    output += f"ttft(Median, P90, P99): {ttft_p[0]:.2f}, {ttft_p[1]:.2f}, {ttft_p[2]:.2f}\n"
    output += f"tpot(Median, P90, P99): {tpot_p[0]:.3f}, {tpot_p[1]:.3f}, {tpot_p[2]:.3f}\n"
    output += f"tps(Median, P90, P99): {tps_p[0]:.1f}, {tps_p[1]:.1f}, {tps_p[2]:.1f}\n"
    output += f"throughput(input, output): {metrics.input_tokens/e2e_duration:.2f}, {metrics.output_tokens/e2e_duration:.2f}\n"
    output += f"num erros: {len(metrics.errors)}\n"
    print(output)
    for ctx in contexts:
        if abs(ctx.output_len - ctx.max_tokens) > 10:
            logger.warning(f"[{ctx.index}] Mismatched output length: expected {ctx.max_tokens}, got {ctx.output_len}")
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(output)

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")
    random.seed(1)
    np.random.seed(1)

    # get providers
    providers = util.get_llm_provider(os.path.dirname(os.path.abspath(__file__)) + "/llm_providers.json")
    if len(providers) == 0:
        raise RuntimeError(f"No valid provider found in the configure file: {config_file}")
    for i in range(min(PRINT_PROVIDERS, len(providers))):
        logger.info(f"Provider[{i}]: {providers[i]}")

    # get samples from dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    norm_prompt_lens = np.rint(np.random.normal(args.input_len, 10, size=args.num_requests)).astype(np.int32)
    norm_output_lens = np.rint(np.random.normal(args.output_len, 6, size=args.num_requests)).astype(np.int32)
    samples = util.load_requests_from_json(tokenizer, args.dataset, args.num_requests, norm_prompt_lens, norm_prompt_lens, norm_output_lens, norm_output_lens)
    logger.info(f"Got {len(samples)} requests")
    contexts = []
    for i in range(len(samples)):
        d = samples[i]
        if i < PRINT_SAMPLES and args.verbose:
            logger.info(f"Request[{i}]: {d[1]} / {d[2]}, {d[0][0: 100]}")
        contexts.append(Context(index=i, prompt=d[0], prompt_len=d[1], max_tokens=d[2]))

    # send requests async
    for provider in providers:
        run_benchmark(provider, contexts, tokenizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # LLM Server
    parser.add_argument("--dataset", type=str, help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--api-kind", type=str, default="completions", choices=["chat", "completions"], help="Can be: chat or completions(default)")
    parser.add_argument("--input-len", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=500)
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of prompts for benckmark.")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--log-file", type=str, help="file to save log information")
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

