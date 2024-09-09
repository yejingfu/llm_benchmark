import argparse
import asyncio
import json
import csv
import os
import sys
import traceback
import time
import warnings
import aiohttp
import requests
import enum
import shlex
from loguru import logger
from tqdm.asyncio import tqdm
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple

import llm_request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

class ApiType(enum.Enum):
    ChatCompletion = enum.auto()
    CompletionStream = enum.auto()

@dataclass
class LlmInputArgs:
    api_type: ApiType
    model: str = ""
    prompt: str = ""
    messages: List[Tuple] = field(default_factory=list)
    stop: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] =  None
    max_tokens: Optional[int] = None
    ## compatible to llm_request.ApiContext
    strict: Optional[bool] = None
    detail: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    peft: Optional[str] = None

def new_input_args(type: ApiType, input: str) -> Optional[LlmInputArgs]:
    result = None
    try:
        obj = json.loads(input)
        if type == ApiType.ChatCompletion and "messages" in obj and len(obj["messages"]) > 0:
            result = LlmInputArgs(type)
            for msg in obj["messages"]:
                if "role" in msg and "content" in msg:
                    result.messages.append(msg)
        elif type == ApiType.CompletionStream and "prompt" in obj and len(obj["prompt"]) > 0:
            result = LlmInputArgs(type)
            result.prompt = obj["prompt"]
        result.model = obj["model"] if "model" in obj else ""
        result.stop = obj["stop"] if "stop" in obj else None
        result.temperature = obj["temperature"] if "temperature" in obj else None
        result.top_p = obj["top_p"] if "top_p" in obj else None
        result.top_k = obj["top_k"] if "top_k" in obj else None
        result.stream = obj["stream"] if "stream" in obj else None
        result.max_tokens = obj["max_tokens"] if "max_tokens" in obj else None
    except Exception as e:
        logger.error(f"Invalid request body: {e}, raw data: {input}")
    return result

def exec_get_method(url, api_key) -> str:
    try:
        headers = {"User-Agent": "LLM API Client"}
        if api_key is not None and api_key != "":
            headers["Authorization"] = "Bearer " + api_key
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.text
        else:
            print(f"Bad GET response: {res}")
    except Exception as e:
        print(f"Exception is raised: {e}")
    return None

async def _chat_chunk_gen(ctx: llm_request.ApiContext, response) -> llm_request.TokenGenerator:
    async for chunk in llm_request.make_sse_chunk_gen(response):
        if chunk.get("choices", []):
            delta = chunk["choices"][0]["delta"]
            delta_content = delta.get("content")
            delta_tool = delta.get("tool_calls")
            if delta_content:
                yield delta_content
            elif delta_tool:
                function = delta_tool[0]["function"]
                name = function.get("name", "").strip()
                if name:
                    yield name
                args = function.get("arguments", "").strip()
                if args:
                    yield args
        usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
        if usage:
            ctx.metrics.input_tokens = usage.get("prompt_tokens")
            ctx.metrics.output_tokens = usage.get("completion_tokens")
            ctx.metrics.provider_queue_time = usage.get("queue_time")
            ctx.metrics.provider_input_time = usage.get("prompt_time")
            ctx.metrics.provider_output_time = usage.get("completion_time")
            ctx.metrics.provider_total_time = usage.get("total_time")

async def _openai_chat_completions(ctx: llm_request.ApiContext, path: str = "/chat/completions"):
    url = ctx.base_url + path
    headers = llm_request.make_headers(auth_token=ctx.api_key)

    kwargs = {"messages": []}
    kwargs["stream_options"] = {"include_usage": True}
    args = ctx.user_data
    if args is not None:
        kwargs["messages"] = args.messages
        if args.stop is not None and len(args.stop) > 0:
            kwargs["stop"] = args.stop
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
        if args.stream is not None:
            kwargs["stream"] = args.stream
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
    data = llm_request.make_openai_chat_body(ctx, **kwargs)
    return await llm_request.post(ctx, url, headers, data, _chat_chunk_gen)


async def _openai_completions(ctx: llm_request.ApiContext, path: str = "/completions"):
    pass

async def send_requests_batch(args: argparse.Namespace, requests: List[LlmInputArgs]):
    timeout = aiohttp.ClientTimeout(total=3600*30)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600*30), connector=aiohttp.TCPConnector(force_close=False)) as session:
        contexts = []
        for i in range(len(requests)):
            req = requsts[i]
            func = _openai_chat_completions if req.api_type == ChatCompletion else _openai_completions
            contexts.append(llm_request.ApiContext(session, i, req.model, func, req, "", [], []))
        num_ctx = len(contexts)
        parallel = args.parallel
        logger.info(f"Start requesting in parallel: {parallel}, total {num_ctx}")
        time_start = time.perf_counter()
        for i in range(0, num_ctx, parallel):
            tasks = [asyncio.create_task(ctx.run()) for ctx in contexts[i : i + parallel]]
            await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - time_start

def main(args):
    if not args.endpoint:
        raise ValueError("Invalid endpoint")
    logger.info(f"LLM serving endpoint: {args.endpoint}")
    models_str = exec_get_method(args.endpoint + "/models", args.api_key)
    if not models_str or args.model_name not in models_str:
        raise RuntimeError(f"Invalid or not supported model: {args.model_name}, serving models: {models_str}")
    if models_str:
        logger.info(f"Supported models: {models_str}")
    requests: List[LlmInputArgs] = []
    if os.path.exists(args.requests_file) and os.path.isfile(args.requests_file):
        logger.info(f"Loading request data from {args.requests_file}")
        with open(args.requests_file, "r") as f:
            try:
                reader = csv.DictReader(f)
                if args.max_num_requests > 0:
                    for i in range(args.max_num_requests):
                        line = next(reader)
                        raw_req = None
                        ts = line["@timestamp"] if "@timestamp" in line else None
                        req_input = None
                        if "msg" in line:
                            raw_req = line["msg"]
                            parts = raw_req.split(", ", 2)
                            if len(parts) == 3 and (parts[0] == "[final] ChatCompletion" or parts[0] == "[final] CompletionStream") and parts[2].startswith("request: "):
                                req_input = new_input_args(ApiType.ChatCompletion if parts[0] == "[final] ChatCompletion" else ApiType.CompletionStream, parts[2][9:])
                        elif "request_body" in line and "request_type" in line:
                            raw_req_body = line["request_body"]
                            raw_req_type = line["request_type"]
                            req_input = new_input_args(ApiType.ChatCompletion if raw_req_type == "ChatCompletion" else ApiType.CompletionStream, raw_req_body)

                        if req_input is None:
                            #logger.warning(f"[{i}] Failed to parse request body: {parts[2][9:]}")
                            logger.warning(f"[{i}] Failed to parse request at: {line['@timestamp']}")
                            continue
                        else:
                            req_input.strict = False
                            req_input.api_key = args.api_key
                            req_input.base_url = args.endpoint
                            requests.append(req_input)
            except StopIteration:
                logger.info("EOS of file")
    if len(requests) == 0:
        logger.error("No valid requests are loaded")
        return
    logger.info(f"{len(requests)} requests are loaded")

    time_start = time.perf_counter()
    asyncio.run(send_requests_batch(requests))
    elapsed = time.perf_counter() - time_start
    logger.info(f"DONE in {elapsed:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read raw client requests and call LLM server one by one")
    parser.add_argument("--endpoint", type=str, help="The host URL of the llm openapi server.", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, help="The model name for completions.")
    parser.add_argument("--api-key", type=str, help="The secret key to connect server.")
    parser.add_argument("--requests-file", type=str, help="The cvs file path which contains original client requests data")
    parser.add_argument("--max-num-requests", type=int, default=100, help="Load maximum requests from the file, default is 100")
    parser.add_argument("--parallel", type=int, default=10, help="The num of requests sent in parallel, default is 10")

    args = parser.parse_args()
    main(args)
