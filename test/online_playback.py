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
PRINT_RESPONSE = 0

class ApiType(enum.Enum):
    ChatCompletion = enum.auto()
    ChatCompletionStream = enum.auto()
    Completion = enum.auto()
    CompletionStream = enum.auto()
    UnknownType = enum.auto()

@dataclass
class LlmInputArgs:
    api_type: ApiType
    raw_model: str = ""
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
    ## response from server
    generated: str = ""

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
        result.raw_model = obj["model"] if "model" in obj else ""
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

def _on_token(ctx: llm_request.ApiContext, token: str):
    ctx.user_data.generated += token

async def _response_chunk_gen(ctx: llm_request.ApiContext, response) -> llm_request.TokenGenerator:
    async for line in response.content:
        result = None
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content == "[DONE]":
                #logger.info(f"[DONE]")
                break
            chunk = json.loads(content)
            if chunk.get("choices", []):
                if "delta" in chunk["choices"][0]: ## chat
                    delta = chunk["choices"][0]["delta"]
                    result = delta.get("content")
                elif "text" in chunk["choices"][0]: ## completions
                    result = chunk["choices"][0]["text"]
                else:
                    logger.warning(f"Unknown choices in response: {chunk['choices']}")
            usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
            if usage:
                ctx.metrics.input_tokens = usage.get("prompt_tokens")
                ctx.metrics.output_tokens = usage.get("completion_tokens")
                ctx.metrics.provider_queue_time = usage.get("queue_time")
                ctx.metrics.provider_input_time = usage.get("prompt_time")
                ctx.metrics.provider_output_time = usage.get("completion_time")
                ctx.metrics.provider_total_time = usage.get("total_time")
        if result is not None:
            yield result

async def _openai_post_message(ctx: llm_request.ApiContext):
    headers = llm_request.make_headers(auth_token=ctx.api_key)
    kwargs = {"stream_options": {"include_usage": True}}
    url = ctx.base_url
    args = ctx.user_data
    if args is not None:
        if args.api_type == ApiType.ChatCompletion:
            url = url + "/chat/completions"
            kwargs["messages"] = args.messages
        else:
            url = url + "/completions"
            kwargs["prompt"] = args.prompt
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
        #else:
        #    kwargs["stream"] = False
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
    data = llm_request.make_openai_chat_body(ctx, **kwargs)
    return await llm_request.post(ctx, url, headers, data, _response_chunk_gen)

async def send_requests_batch(args: argparse.Namespace, req_list: List[LlmInputArgs]):
    timeout = aiohttp.ClientTimeout(total=3600*30)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600*30), connector=aiohttp.TCPConnector(force_close=False)) as session:
        contexts = []
        for i in range(len(req_list)):
            req = req_list[i]
            ctx = llm_request.ApiContext(session, i, req.raw_model, _openai_post_message, req, "", [], [])
            ctx.user_data = req
            contexts.append(ctx)
        num_ctx = len(contexts)
        parallel = args.parallel
        logger.info(f"Start requesting in parallel: {parallel}, total {num_ctx}")
        time_start = time.perf_counter()
        for i in range(0, num_ctx, parallel):
            tasks = [asyncio.create_task(ctx.run(_on_token)) for ctx in contexts[i : i + parallel]]
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
    req_list : List[LlmInputArgs] = []
    def _get_api_type(s:str) -> ApiType:
        t = ApiType.UnknownType
        if s.startswith("ChatCompletionStream"):
            t = ApiType.ChatCompletionStream
        elif s.startswith("ChatCompletion"):
            t = ApiType.ChatCompletion
        elif s.startswith("CompletionStream"):
            t = ApiType.CompletionStream
        elif s.startswith("Completion"):
            t = ApiType.Completion
        return t
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
                        t = ApiType.UnknownType
                        b = None
                        if "msg" in line:
                            raw_req = line["msg"]
                            parts = raw_req.split(", ", 2)
                            if len(parts) == 3 and parts[0].startswith("[final] ") and parts[2].startswith("request: "):
                                t = _get_api_type(parts[0][8:])
                                b = parts[2][9:]
                        elif "request_body" in line and "request_type" in line:
                            t = _get_api_type(line["request_type"])
                            b = line["request_body"]

                        if t == ApiType.UnknownType:
                            logger.warning(f"[{i}]: unknown api type: {parts[0][8:]}, body: {b}, at {line['@timestamp']}")
                        #else:
                        elif t == ApiType.ChatCompletion:
                            req_input = new_input_args(t, b)
                            req_input.model = args.model_name
                            req_input.strict = False
                            req_input.api_key = args.api_key
                            req_input.base_url = args.endpoint
                            req_list.append(req_input)
            except StopIteration:
                logger.info("EOS of file")
    if len(req_list) == 0:
        logger.error("No valid requests are loaded")
        return
    logger.info(f"{len(req_list)} requests are loaded")

    time_start = time.perf_counter()
    asyncio.run(send_requests_batch(args, req_list))
    elapsed = time.perf_counter() - time_start
    logger.info(f"DONE in {elapsed:.3f} seconds")
    if PRINT_RESPONSE > 0:
        num = min(PRINT_RESPONSE, len(req_list))
        logger.info(f"Print the generated content about the first {num} requests")
        for i in range(num):
            logger.info(f"[{i}] type: {req_list[i].api_type}, model: {req_list[i].raw_model}, generated: {req_list[i].generated}")

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
