import aiohttp
import argparse
import asyncio
import json
import numpy as np
import os
import time
from urllib.parse import urlparse
from loguru import logger
from async_request_sender import Context, AysncRequestSender
import util


async def run_profile(args: argparse.Namespace):
    logger.info(args)
    if args.endpoint is None:
        raise ValueError("Invalid endpoint")
    base_url = args.endpoint
    parsed_url = urlparse(args.endpoint)
    if parsed_url.path:
        base_url = base_url[:len(base_url)-len(parsed_url.path)]
    sample_path = os.path.dirname(os.path.abspath(__file__)) + "/stability_samples.json"
    logger.info(f"Load prompt samples from {sample_path}")
    with open(sample_path, "r") as f:
        data = json.load(f)
        data = data["data"]
        for d in data:
            prompt_len = d["prompt_len"]
            if prompt_len >= args.input_len
                prompt = d["prompt"]
                max_tokens = prompt_len // 10
                break
    if prompt is None:
        raise RuntimeError("Invalid prompt")
    logger.info(f"Got prompt(len={prompt_len}, {max_tokens}): {prompt}")
    model_name = util.get_model(args.endpoint + "/models")
    logger.info(f"Serving model name: {model_name}")

    sender = AysncRequestSender(args.endpoint, model_name, None, None, False, True, args.verbose)
    extra = {}
    start_time = time.perf_counter()
    # start profile
    ctx = Context(prompt="start profile")
    await sender.do_post_request(base_url+"/start_profile", ctx, False, extra)
    if ctx.error:
        logger.error(f"Error of start profile: {ctx.error}")
    # real request
    req_ctx = Context(prompt=prompt, prompt_len=prompt_len, max_tokens=max_tokens)
    await sender.do_post_request(args.endpoint+"/completions", req_ctx, False, extra)
    # stop profile
    ctx = Context(prompt="stop profile")
    await sender.do_post_request(base_url+"/stop_profile", ctx, False, extra)
    if ctx.error:
        logger.error(f"Error of stop profile: {ctx.error}")
    e2e_duration = time.perf_counter() - start_time
    logger.info(f"==== Finished in {e2e_duration} seconds =====")
    print(f"generated({req_ctx.output_len}): {req_ctx.generated}")
    print(f"e2e_latency: {req_ctx.e2e_latency}")
    print(f"error: {req_ctx.error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile vllm server with request."
    )
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--input-len", type=int, default=5000, help="The sequence input length, default is 5000")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
    args = parser.parse_args()
    asyncio.run(run_profile(args))

