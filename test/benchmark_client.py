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
from typing import List, Tuple, Union, Optional, Dict
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from dataset_sampler import DatasetSampler
from async_request_sender import Response, InputParameter, AysncRequestSender

REQUEST_RESPONSES: List[Response] = []

SYS_PROMPT="""
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
TMP_TEST_SHARED_PROMPT = False

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    sender = AysncRequestSender(args.backend, args.base_url, args.api_key, args.endpoint_models, args.endpoint_chat, args.endpoint_completion, SYS_PROMPT if args.add_system_prompt else None)
    if not args.ignore_check:
        if not sender.check_health(10):
            logger.error(f"Failed to check the healthy of the inference server")
            return
        str_models = sender.get_models()
        if str_models is None:
            logger.error("No valid models supported from server")
            return
        logger.info(f"[Model list]: {str_models}")
        json_models = json.loads(str_models)
        model_list = []
        for model in json_models['data']:
            model_list.append(model['id'])
        logger.info(f"Supported models: {model_list}")
        if args.model not in model_list:
            logger.error(f"The LLM server does not support this model: {args.model}")

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=args.use_fast)
    sampler = DatasetSampler(args.dataset)
    requests_warmup, requests_test = sampler.sample_requests(
        args.num_warmup_requests,
        args.num_benchmark_requests,
        tokenizer,
        SYS_PROMPT if args.add_system_prompt and args.api_kind == "completions" else None,
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

    parameters = InputParameter(model=args.model, n=args.n, best_of=args.best_of, use_beam_search=args.use_beam_search, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, repetition_penalty=args.repetition_penalty,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k if args.do_sample else None, max_tokens=args.fixed_output_len, ignore_eos=None, stream=args.stream
    )

    if args.dry_run:
        logger.info("======== basic input parameters ======")
        logger.info(f"{parameters.to_dict()}")
        num = 10 if len(requests_test) > 10 else len(requests_test)
        logger.info(f"========================= print the first {num} requests from test set ===============================")
        for i in range(num):
            prompt, in_len, out_len = requests_test[i]
            logger.info(f"request[{i}]: ({in_len}, {out_len}): {prompt}")
        logger.info("\n\n")
        return

    if TMP_TEST_SHARED_PROMPT:
        ## repeat per 10 requests
        requests_test_tmp = []
        num = len(requests_test)
        for i in range(num):
            requests_test_tmp.append(requests_test[int(i/10)])
        requests_test = requests_test_tmp

    # post requests
    for phase, input_requests in zip(("Warmup", "Benchmark"), (requests_warmup, requests_test)):
        if len(input_requests) == 0:
            continue
        start_time = time.perf_counter()
        asyncio.run(sender.post_batch_requests_async(args.max_concurrent_requests, input_requests, parameters, args.api_kind == "chat"))
        end_time = time.perf_counter()
        if phase == "Benchmark":
            sender.dump_response_stats(tokenizer, args.stream, end_time - start_time, args.warn_dismatch_output_len, args.log_file)

def simple_verify_args(args):
    assert not args.use_beam_search, "do not support benchmark beam search now."
    assert (args.best_of is None or args.best_of == 1 and args.n == 1), "do not support benchmark best_of and n now."
    assert (args.presence_penalty == 0.0 and args.frequency_penalty == 0.0 and args.repetition_penalty == 1.0), "do not support benchmark penalty policies now."
    if args.do_sample:
        assert (args.temperature > 0.0), "temperature must be greater than 0.0 when do_sample is True."
    else:
        assert args.top_k == 1 and args.top_p == 1.0 and args.temperature == 0.0
    if not args.stream:
        logger.warning("The --stream is not set, are you sure run in non-stream mode?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # LLM Server
    parser.add_argument("--backend", type=str, help="The backend e.g. vllm, trtllm")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, help="The api key to call commercial inference API")
    parser.add_argument("--api-kind", type=str, default="completions", choices=["chat", "completions"], help="Call char-completions API or completions API")
    parser.add_argument("--endpoint-models", type=str, help="The endpoint to call server health checking")
    parser.add_argument("--endpoint-chat", type=str, help="The endpoint to call chat API")
    parser.add_argument("--endpoint-completion", type=str, help="The endpoint to call completion API")
    # input parameters
    parser.add_argument("--model", type=str, default="default", help="The model name")
    parser.add_argument("--n", type=int, default=1, help="How many sequences to generate for each prompt.")
    parser.add_argument("--best-of", type=int, default=None, help="Generates `best_of` sequences per prompt and returns the top `n` results, with the default value of `n` being one.")
    parser.add_argument("--use-beam-search", action="store_true", default=None)
    parser.add_argument("--do-sample", action="store_true", help="if this value is set, use top_k")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0, help="temperature parameter")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter")
    parser.add_argument("--top_k", type=int, default=1, help="top-k parameter")
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    # test data sampling
    parser.add_argument("--sampling-policy", type=str, default="nature", choices=["nature", "fixed", "normal"])
    parser.add_argument("--max_turns", type=int, default=4096)
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
    # press test setting
    parser.add_argument("--num-warmup-requests", type=int, default=100, help="Number of prompts for warmup.")
    parser.add_argument("--num-benchmark-requests", type=int, default=8000, help="Number of prompts for benckmark.")
    parser.add_argument("--max-concurrent-requests", type=int, default=400)
    parser.add_argument("--ramp_up_period", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--trust-remote-code", action="store_true", help="trust remote code from huggingface")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    parser.add_argument("--ignore-check", action="store_true", help="do not check health and model validity before send requests")
    parser.add_argument("--log-file", type=str, default="", help="file to save log information")
    parser.add_argument("--dry-run", action="store_true", help="Don't run the benchmark really, only print some message for debugging")
    args = parser.parse_args()
    simple_verify_args(args)
    main(args)

