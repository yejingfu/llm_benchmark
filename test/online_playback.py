import argparse
import asyncio
import json
import re
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
import numpy as np
from loguru import logger
from tqdm.asyncio import tqdm
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple
from transformers import AutoTokenizer

import llm_request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
PRINT_VERBOSE = True
PRINT_STREAM_TOKEN = False

class ApiType(enum.Enum):
    ChatCompletion = enum.auto()
    ChatCompletionStream = enum.auto()
    Completion = enum.auto()
    CompletionStream = enum.auto()
    UnknownType = enum.auto()

@dataclass
class LlmInputArgs:
    api_type: ApiType = ApiType.UnknownType
    index: int = 0
    timestamp: str = ""
    raw_model: str = ""
    model: str = ""

    prompt: Optional[str] = field(default=None)
    messages: Optional[List[Tuple]] = field(default=None)
    frequency_penalty: Optional[float] = field(default=None)
    repetition_penalty: Optional[float] = field(default=None)
    presence_penalty: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    min_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    max_tokens: Optional[int] = field(default=None)
    n: Optional[int] = field(default=None)
    stop: Optional[List[str]] = field(default=None)
    ignore_eos: Optional[bool] = field(default=None)
    stream: Optional[bool] =  field(default=None)
    prompt_len: Optional[int] = field(default=None)
    ## compatible to llm_request.ApiContext
    strict: Optional[bool] = None
    detail: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    peft: Optional[str] = None
    ## response from server
    generated: str = ""

    def get_request_body(self):
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "stream": self.stream
        }

def new_input_args(type: ApiType, input: str) -> Optional[LlmInputArgs]:
    input_args = None
    try:
        obj = json.loads(input)
        if type == ApiType.UnknownType:
            if "messages" in obj:
                type = ApiType.ChatCompletionStream
                input_args = LlmInputArgs(type)
                input_args.messages = []
                for msg in obj["messages"]:
                    if "role" in msg and "content" in msg:
                        input_args.messages.append(msg)
            elif "prompt" in obj:
                type = ApiType.CompletionStream
                input_args = LlmInputArgs(type)
                input_args.prompt = obj["prompt"]
            else:
                return None
        elif type == ApiType.ChatCompletion or type == ApiType.ChatCompletionStream:
            if "messages" in obj and len(obj["messages"]) > 0:
                input_args = LlmInputArgs(type)
                for msg in obj["messages"]:
                    if "role" in msg and "content" in msg:
                        input_args.messages.append(msg)
        elif type == ApiType.Completion or type == ApiType.CompletionStream:
            if "prompt" in obj and len(obj["prompt"]) > 0:
                input_args = LlmInputArgs(type)
                input_args.prompt = obj["prompt"]
        for k in obj:
                v = obj[k]
                if k == "model":
                    input_args.raw_model = v
                elif k == "frequency_penalty":
                    input_args.frequency_penalty = v
                elif k == "repetition_penalty":
                    input_args.repetition_penalty = v
                elif k == "presence_penalty":
                    input_args.presence_penalty = v
                elif k == "temperature":
                    input_args.temperature = v
                elif k == "top_p":
                    input_args.top_p = v
                elif k == "min_p":
                    input_args.min_p = v
                elif k == "top_k":
                    input_args.top_k = v
                elif k == "max_tokens":
                    input_args.max_tokens = v
                elif k == "n":
                    input_args.n = v
                elif k == "stop":
                    input_args.stop = v
                elif k == "ignore_eos":
                    input_args.ignore_eos = v
                elif k == "stream":
                    input_args.stream = v
                else:
                    if k not in ["prompt", "messages"]:
                        logger.warning(f"not support parameter: {k}: {v}")
    except Exception as e:
        logger.error(f"Invalid request body: {e}, raw data: {input}")
    return input_args

def load_inputs_from_csv(file_path:str, num_reqs:int, req_ids:List[int]):
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

    def _decode_unicode_strings(text):
        if len(text) > 0 and text[0] == '\ufeff':
            text = text[1:]
        pattern = r'\\u[\da-fA-F]{4}'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                decoded = bytes(match, 'ascii').decode('unicode_escape')
                text = text.replace(match, decoded)
            except UnicodeDecodeError:
                pass
        return text
    req_list = []
    if os.path.exists(file_path) and os.path.isfile(file_path):
        logger.info(f"Loading request data from {file_path}")
        with open(file_path, "r") as f:
            try:
                reader = csv.DictReader(f)
                req_idx = 0
                line = next(reader)
                while line and (num_reqs <=0 or req_idx < num_reqs):
                    line2 = {_decode_unicode_strings(k):v for k,v in line.items()}
                    line = line2
                    ts = line["@timestamp"] if "@timestamp" in line else ""
                    body = None
                    api_type = ApiType.UnknownType
                    if "msg" in line:
                        raw_req = line["msg"]
                        parts = raw_req.split(", ", 2)
                        if len(parts) == 3 and parts[0].startswith("[final] ") and parts[2].startswith("request: "):
                            api_type = _get_api_type(parts[0][8:])
                            body = parts[2][9:]
                    elif "request_body" in line:
                        api_type = _get_api_type(line["request_type"]) if "request_type" in line else api_type
                        body = line["request_body"]
                    if body is not None and (req_ids is None or req_idx in req_ids):
                        body = re.sub(r'\\(.)', r'\1', body.strip())
                        #print(f"==== raw request body[{req_idx}]: {body}")
                        req_input = new_input_args(api_type, body)
                        if req_input is not None:
                            req_input.index = req_idx
                            req_input.timestamp = ts
                            req_list.append(req_input)
                    line = next(reader)
                    req_idx += 1
            except StopIteration:
                logger.info("EOS of file")
    print(f"=== len: {len(req_list)}")
    return req_list

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
    if PRINT_STREAM_TOKEN:
        print(token, end="", flush=True)

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

async def _openai_post_message(ctx: llm_request.ApiContext, phase: llm_request.RequestPhase, response = None):
    args = ctx.user_data
    assert args is not None
    if phase == llm_request.RequestPhase.End:
        if PRINT_VERBOSE:
            logger.info(f"Finish request[{args.index}]: {args.api_type}, {args.prompt_len} / {args.max_tokens}")
        return None

    if PRINT_VERBOSE:
        logger.info(f"Send request[{args.index}]: {args.api_type}, {args.prompt_len} / {args.max_tokens}")
    headers = llm_request.make_headers(auth_token=ctx.api_key)
    url = ctx.base_url
    data = {
        "model": ctx.model or None,
        "max_tokens": ctx.max_tokens,
        "temperature": ctx.temperature,
        "stream_options": {"include_usage": True},
    }

    params = args.get_request_body()
    for k, v in params.items():
        if v is not None:
            data[k] = v
    data["stream"] = True ## ALWAYS set stream to TRUE  !!!!!
    #data["stream"] = (args.api_type == ApiType.ChatCompletionStream or args.api_type == ApiType.CompletionStream) or None
    #if data["stream"] != True:
    #    data.pop("stream")
    if args.api_type == ApiType.ChatCompletion or args.api_type == ApiType.ChatCompletionStream:
        url = url + "/chat/completions"
        #del data["prompt"]
        #data["messages"] = args.messages
        #data["stream"] = args.api_type == ApiType.ChatCompletionStream
    else:
        url = url + "/completions"
        #del data["messages"]
        #data["prompt"] = args.prompt
        #data["stream"] = args.api_type == ApiType.CompletionStream

    return await llm_request.post(ctx, url, headers, data, _response_chunk_gen)

async def send_requests_batch(args: argparse.Namespace, req_list: List[LlmInputArgs]) -> List[llm_request.ApiContext]:
    contexts = []
    timeout = aiohttp.ClientTimeout(total=3600*30)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600*30), connector=aiohttp.TCPConnector(force_close=False)) as session:
        pbar = tqdm(total=len(req_list))
        for i in range(len(req_list)):
            req = req_list[i]
            ctx = llm_request.ApiContext(session, i, req.raw_model, _openai_post_message, req, "", [], [])
            ctx.user_data = req
            contexts.append(ctx)
        num_ctx = len(contexts)
        parallel = args.parallel
        logger.info(f"Start requesting in parallel: {parallel}, total {num_ctx}")
        for i in range(0, num_ctx, parallel):
            tasks = [asyncio.create_task(ctx.run(_on_token)) for ctx in contexts[i : i + parallel]]
            await asyncio.gather(*tasks)
            pbar.update(parallel)
        pbar.close()

    return contexts

def save_result_csv(file_name:str, contexts: List[llm_request.ApiContext]):
    file = open(file_name, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    header = ["index","ttft","tps","input-len","output-len","total-time","p-queue-time","p-input-time","p-output-time","p-total-time","model","timestamp"]
    writer.writerow(header)
    bad_metrics = []
    for i in range(len(contexts)):
        m = contexts[i].metrics
        in_args = contexts[i].user_data
        if not m.error:
            writer.writerow([i, round(m.ttft,2), round(m.tps), m.input_tokens, m.output_tokens, round(m.total_time or 0,2), round(m.provider_queue_time or 0,2), round(m.provider_input_time or 0,2), round(m.provider_output_time or 0,2), round(m.provider_total_time or 0,2), m.model, in_args.timestamp])
        else:
            bad_metrics.append(in_args.timestamp + ", " + m.error)
    file.close()
    if len(bad_metrics) > 0:
        with open(file_name + ".err", "w") as f:
            for m in bad_metrics:
                f.write(m+"\n")

    logger.info(f"got good metrics: {len(contexts) - len(bad_metrics)}, bad metrics: {len(bad_metrics)}")

def main(args):
    if not args.endpoint:
        raise ValueError("Invalid endpoint")
    logger.info(f"LLM serving endpoint: {args.endpoint}")
    if not args.model_name:
        res = requests.get(args.endpoint + "/models")
        model_list = res.json().get("data", [])
        args.model_name = model_list[0]["id"] if model_list else None
    logger.info(f"Model name: {args.model_name}")
    if not args.model_name:
        raise ValueError(f"Invalid model name")
    if args.parallel is None or args.parallel == 0:
        args.parallel = 10
    logger.info(f"Parallel: {args.parallel}")
    req_ids = None
    if args.requests_ids:
        req_ids = args.requests_ids.split(",")
        req_ids = [int(x) for x in req_ids]
        if len(req_ids) == 0:
            req_ids = None
    req_list = load_inputs_from_csv(args.requests_file, args.max_num_requests, req_ids)
    if len(req_list) == 0:
        logger.error("No valid requests are loaded")
        return
    for req in req_list:
        req.api_key = args.api_key
        req.base_url = args.endpoint
        req.model = args.model_name
    logger.info(f"{len(req_list)} requests are loaded")

    ## print the promt len and max_tokens
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for i in range(len(req_list)):
            prompt_len = len(tokenizer.encode(req_list[i].prompt)) if req_list[i].prompt else 0
            prompt_len += sum([len(tokenizer.encode(f"{x}")) for x in req_list[i].messages])
            req_list[i].prompt_len = prompt_len
    if args.print_requests:
        for i in range(len(req_list)):
            print(f"[{i}]: {req_list[i]}")

    ## send requests in parallel
    time_start = time.perf_counter()
    contexts = asyncio.run(send_requests_batch(args, req_list))
    elapsed = time.perf_counter() - time_start
    logger.info(f"DONE in {elapsed:.3f} seconds")

    ## print statistics
    ttft = []
    tps = []
    e2e_latency = []
    input_len = []
    output_len = []
    errors = []
    for c in contexts:
        if c.metrics.error:
            errors.append(c.metrics.error)
        else:
            ttft.append(c.metrics.ttft)
            tps.append(c.metrics.tps)
            e2e_latency.append(c.metrics.total_time)
            input_len.append(c.metrics.input_tokens)
            output_len.append(c.metrics.output_tokens)
    if len(ttft) > 0:
        PERCENTILES = [50, 90, 99]
        ttft_p = [round(float(x), 2) for x in np.percentile(ttft, PERCENTILES)]
        tps_p = [round(x) for x in np.percentile(tps, PERCENTILES)]
        e2e_latency_p = [round(float(x), 2) for x in np.percentile(e2e_latency, PERCENTILES)]
        input_len_p = np.percentile(input_len, PERCENTILES)
        output_len_p = np.percentile(output_len, PERCENTILES)
        logger.info(f"TTFT(Median, P90, P99): {ttft_p}")
        logger.info(f"TPS(Median, P90, P99): {tps_p}")
        logger.info(f"E2E Latency(Median, P90, P99): {e2e_latency_p}")
        logger.info(f"input len(Median, P90, P99): {input_len_p}")
        logger.info(f"output len(Median, P90, P99): {output_len_p}")
    elif len(errors) > 0:
        for i in range(len(errors)):
            logger.warning(f"ERROR[{i}]: {errors[i]}")
    if args.print_response > 0:
        num = min(args.print_response, len(req_list))
        logger.info(f"Print the generated content about the first {num} requests")
        for i in range(num):
            print(f"\n=============\nRequest[{i}] type: {req_list[i].api_type}, model: {req_list[i].raw_model}\n")
            print(f"\n###Request: {req_list[i].get_request_body()}")
            print(f"\n###Response: {req_list[i].generated}")
    if args.dump is not None:
        save_result_csv(args.dump, contexts)

if __name__ == "__main__":
    ## example:
    ### python test/online_playback.py --endpoint http://localhost:18011/v1 --model-name llama31-8b --requests-file test/data/microsoftwizardlm-2-8x22b-errors.csv --max-num-requests 10 --parallel 5 --dump result_playback.csv
    parser = argparse.ArgumentParser(description="Read raw client requests and call LLM server one by one")
    parser.add_argument("--endpoint", type=str, help="The host URL of the llm openapi server.", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, help="The model name for completions, if not set, call endpoint to query.")
    parser.add_argument("--tokenizer", type=str, help="Optional, if set, use it to calualate the input lengh")
    parser.add_argument("--api-key", type=str, help="The secret key to connect server.")
    parser.add_argument("--requests-file", type=str, help="The cvs file path which contains original client requests data")
    parser.add_argument("--requests-ids", type=str, help="Pick the specific requests from file for testing, start from 0, seperated by comma")
    parser.add_argument("--max-num-requests", type=int, default=100, help="Load maximum requests from the file, default is 100")
    parser.add_argument("--parallel", type=int, default=10, help="The num of requests sent in parallel, default is 10")
    parser.add_argument("--print-requests", action="store_true", help="Print all raw requests")
    parser.add_argument("--print-response", type=int, default=0, help="The num of the response to print, default is 0")
    parser.add_argument("--dump", type=str, help="The file to save the output data")

    args = parser.parse_args()
    main(args)
