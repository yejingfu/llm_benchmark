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
from dataset_sampler import DatasetSampler
from async_request_sender import Response, AysncRequestSender

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

def main(args: argparse.Namespace):
    logger.info("=========== arguments =======================")
    logger.info(args)
    logger.info("\n\n")



    sender = AysncRequestSender(args.backend, args.base_url, args.api_key, args.endpoint_models, args.endpoint_chat, args.endpoint_completion)
    if not sender.check_health(10):
        logger.error(f"Failed to check the healthy of the inference server")
    str_models = sender.get_models()
    if str_models is None:
        logger.error("No valid models supported from server")
        return
    logger.info(f"[Model list]: {str_models}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=args.use_fast)
    sampler = DatasetSampler(args.dataset)
    requests_warmup, requests_test = sampler.sample_requests(
        args.num_warmup_requests,
        args.num_benchmark_requests,
        tokenizer,
        args.add_system_prompt,
        args.sampling_policy,
        max_turns=args.max_turns,
        min_prompt_len=args.min_prompt_len,
        max_prompt_len=args.max_prompt_len,
        min_output_len=args.min_output_len,
        max_prompt_output_len=args.max_prompt_output_len,
        fixed_prompt_len=args.fixed_prompt_len,
        fixed_output_len=args.fixed_output_len,
        max_seq_len=args.max_seq_len,
        prompt_len_mean=args.prompt_len_mean,
        prompt_len_std=args.prompt_len_std,
        output_len_mean=args.output_len_mean,
        output_len_std=args.output_len_std,
    )

    if args.dry_run:
        num = 10 if len(requests_test) > 10 else len(requests_test)
        logger.info(f"========================= print the first {num} requests from test set ===============================")
        for i in range(num):
            prompt, in_len, out_len = requests_test[i]
            logger.info(f"request[{i}]: ({in_len}, {out_len}): {prompt}")
        logger.info("\n\n")
        return

    # post requests
    for phase, input_requests in zip(("Warmup", "Benchmark"), (requests_warmup, requests_test)):
        if len(input_requests) == 0:
            continue
        start_time = time.perf_counter()
        asyncio.run(sender.post_batch_requests_async(
            args.max_concurrent_requests, input_requests, args.model, args.n, args.best_of, args.use_beam_search, args.do_sample, args.presence_penalty, args.frequency_penalty, args.repetition_penalty,
            args.temperature, args.top_p, args.top_k if args.top_k > 0 else tokenizer.vocab_size, args.stream, tokenizer.eos_token_id,
        ))
        end_time = time.perf_counter()
        if phase == "Benchmark":
            sender.dump_response(tokenizer, args.stream, args.gpus, end_time - start_time, args.log_file)

def simple_verify_args(args):
    assert not args.use_beam_search, "do not support benchmark beam search now."
    assert (args.best_of == 1 and args.n == 1), "do not support benchmark best_of and n now."
    assert (args.presence_penalty == 0.0 and args.frequency_penalty == 0.0 and args.repetition_penalty == 1.0), "do not support benchmark penalty policies now."
    if args.do_sample:
        assert (args.temperature > 0.0), "temperature must be greater than 0.0 when do_sample is True."
    else:
        assert args.top_k == 1 and args.top_p == 1.0 and args.temperature == 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--backend", type=str, help="The backend e.g. vllm, trtllm")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, help="The api key to call commercial inference API")
    parser.add_argument("--endpoint-models", type=str, help="The endpoint to call server health checking")
    parser.add_argument("--endpoint-chat", type=str, help="The endpoint to call chat API")
    parser.add_argument("--endpoint-completion", type=str, help="The endpoint to call completion API")
    parser.add_argument("--dataset", type=str, help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--image", type=str, help="The inference image, if set this value, run docker container from this image before testing")
    parser.add_argument("--model", type=str, default="default", help="The model name")
    parser.add_argument("--n", type=int, default=1, help="How many sequences to generate for each prompt.")
    parser.add_argument("--best-of", type=int, default=1, help="Generates `best_of` sequences per prompt and returns the top `n` results, with the default value of `n` being one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sampling-policy", type=str, default="nature", choices=["nature", "fixed", "normal"])
    parser.add_argument("--max_turns", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--generate-length", type=int, default=64)
    parser.add_argument("--min_prompt_len", type=int, default=4)
    parser.add_argument("--min_output_len", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--max_prompt_output_len", type=int, default=4096)
    parser.add_argument("--fixed_prompt_len", type=int, default=3500)
    parser.add_argument("--fixed_output_len", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--prompt_len_mean", type=int, default=550)
    parser.add_argument("--prompt_len_std", type=int, default=150)
    parser.add_argument("--output_len_mean", type=int, default=150)
    parser.add_argument("--output_len_std", type=int, default=20)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter")
    parser.add_argument("--top_k", type=int, default=1, help="top-k parameter")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature parameter")
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--decode-strategy", type=str, required=False, choices=[])
    parser.add_argument("--quentize", type=str, required=False, choices=[])
    parser.add_argument("--num-warmup-requests", type=int, default=100, help="Number of prompts for warmup.")
    parser.add_argument("--num-benchmark-requests", type=int, default=8000, help="Number of prompts for benckmark.")
    parser.add_argument("--max-concurrent-requests", type=int, default=400)
    parser.add_argument("--ramp_up_period", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", help="trust remote code from huggingface")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--pad-requests", action="store_true", help="pad the requests repeatedly when the number of sampled requests is less than expected")
    parser.add_argument("--warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    parser.add_argument("--log-file", type=str, default="", help="file to save log information")
    parser.add_argument("--generated-log-file", type=str, default=None, help="file to save generated text")
    parser.add_argument("--gpus", type=int, default=8, help="The total num of GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Don't run the benchmark really, only print some message for debugging")
    args = parser.parse_args()
    #args.prompt_len_mean = args.context_length
    #args.output_len_mean = args.generate_length

    simple_verify_args(args)
    main(args)

