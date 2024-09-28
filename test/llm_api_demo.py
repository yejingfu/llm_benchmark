import argparse
import asyncio
import json
import os
import sys
import traceback
import time
import aiohttp
import requests
from urllib.parse import urlparse
from loguru import logger
from tqdm.asyncio import tqdm
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional, Tuple

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

"""
Example of APIs:
    "version": "/version",
    "health": "/health",
    "models": "/v1/models",
    "completions": "/v1/completions",
    "chat-completions": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "compatible-embeddings": "/compatible-mode/v1/embeddings"
"""

@dataclass
class CompletionContext:
    index: int = field(default=0)
    # input
    model: str = field(default="")
    prompt: str = field(default="")
    prompt_len: int = field(default=0)
    max_tokens: int = field(default=10)
    stream: Optional[bool] = field(default=False)
    ignore_eos: Optional[bool] = field(default=False)
    # output
    error: Optional[str] = field(default=None)
    generated: str = field(default="")
    output_len: int = field(default=0)
    e2e_latency: float = field(default=0)
    decode_latency: float = field(default=0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)
    itl: Optional[float] = field(default=None)
    tps: float = field(default=0)

@dataclass
class EmbeddingOutput:
    embeddings: List[float] = field(default_factory=list)
    prompt_len: int = 0
    total_len: int = 0
    err: str = ""

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def print_chunk_streamly(count: int, data):
    cur = None
    if "choices" in data:
        if "text" in data["choices"][0]:
            cur = f"{data['choices'][0]['text']}"
        elif "message" in data["choices"][0]:
            cur = f"{data['choices'][0]['message']['content']}"
        elif "delta" in data["choices"][0]:
            if "content" in data['choices'][0]['delta']:
                cur = f"{data['choices'][0]['delta']['content']}"
    elif "text" in data:
        cur = f"{data['text']}"

    if cur is not None:
        sys.stdout.write(cur)
        sys.stdout.flush()

async def async_request_openai_completions(ctx: CompletionContext, url: str, api_key: Optional[str] = None, pbar: Optional[tqdm] = None):
    payload = {
        "model": ctx.model,
        "max_tokens": ctx.max_tokens,
    }
    if ctx.stream is not None:
        payload["stream"] = ctx.stream
    if ctx.ignore_eos is not None:
        payload["ignore_eos"] = ctx.ignore_eos
    chat = "/chat/completions" in url
    if os.path.isfile(ctx.prompt):
        print(f"Load prompt from {ctx.prompt}")
        with open(ctx.prompt, "r") as f:
            data = f.read()
            if ctx.prompt.endswith(".json") or ctx.prompt.endswith(".jsonl"):
                data = json.loads(data)
                if "prompt" in data or "messages" in data:
                    for k in data:
                        payload[k] = data[k]
                else:
                    raise ValueError(f"Invalid content in file {ctx.prompt}")
            else:
                if chat:
                    payload["messages"] = []
                    payload["messages"].append({"role": "user", "content": data})
                else:
                    payload["prompt"] = data
    else:
        if chat:
            payload["messages"] = []
            payload["messages"].append({"role": "user", "content": args.prompt})
        else:
            payload["prompt"] = args.prompt

    ## more parameters
    #payload["temperature"] = 0.2
    #payload["top_k"] = 5
    #payload["frequency_penalty"] = 0.5
    #payload["presence_penalty"] = 0.1
    #payload["max_tokens"] = 30

    headers = {"User-Agent": "LLM API Client"}
    #headers["accept"] = "application/json"
    #headers["content-type"] = "application/json"
    if api_key is not None and api_key != "":
        headers["Authorization"] = "Bearer " + api_key

    print(f"URL: {url}, Payload: {payload}")
    st_start = time.perf_counter()
    most_recent_timestamp = st_start
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            iter = 0
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        iter += 1
                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == ": OPENROUTER PROCESSING":
                            continue
                        if chunk == "[DONE]":
                            ctx.e2e_latency = time.perf_counter() - st_start
                        else:
                            data = json.loads(chunk)
                            if ctx.stream:
                                print_chunk_streamly(iter, data)
                            text = None
                            if "choices" in data:
                                c0 = data["choices"][0]
                                if "text" in c0:
                                    text = c0["text"]
                                elif "message" in c0 and "content" in c0["message"]:
                                    text = c0["message"]["content"]
                                elif "delta" in c0 and "content" in c0["delta"]:
                                    text = c0["delta"]["content"]
                            elif "text" in data:
                                text = data["text"]
                            if text is not None:
                                timestamp = time.perf_counter()
                                # First token
                                if ctx.ttft is None:
                                    ctx.ttft = timestamp - st_start
                                # Decoding phase
                                elif data.get("usage", None) is None:
                                    ctx.itl =  [] if ctx.itl is None else ctx.itl
                                    ctx.itl.append(timestamp - most_recent_timestamp)
                                most_recent_timestamp = timestamp
                                ctx.generated += text
                    if not ctx.stream:
                        print(ctx.generated)
                    if ctx.e2e_latency < 0.0001:
                        ctx.e2e_latency = time.perf_counter() - st_start
                else:
                    details = await response.text()
                    ctx.error = f"status code {response.status}({response.reason}).\n\t Details: {details}."
        except Exception:
            exc_info = sys.exc_info()
            ctx.error = "".join(traceback.format_exception(*exc_info))
    if pbar:
        pbar.update(1)

## reference: https://help.aliyun.com/zh/model-studio/developer-reference/embedding-interfaces-compatible-with-openai
async def async_generate_embeddings(url: str, input: str, model_name: str, api_key: str = None) -> EmbeddingOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "input": input,
            "model": model_name,
            "encoding_format": "float"
        }
        headers = {"User-Agent": "LLM API Client"}
        if api_key is not None and api_key != "":
            headers["Authorization"] = "Bearer " + api_key
        output: EmbeddingOutput = None
        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        data = json.loads(chunk_bytes)
                        if "data" in data and len(data["data"]) > 0:
                            e = data["data"][0]["embedding"]
                            output = EmbeddingOutput(embeddings=data["data"][0]["embedding"], prompt_len=data["usage"]["prompt_tokens"], total_len=data["usage"]["total_tokens"])

                else:
                    logger.error(f"Failed to generate embedddings, error: {response.status}, {response.reason}")
        except Exception as ex:
            logger.error(f"Exception raised when generate embeddings: {ex}")
        return output

def check_health(url: str, ) -> bool:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        return True
    else:
        return False

def get_version(url: str):
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
        return res.json().get("version")
    else:
        raise RuntimeError(f"Failed to get version({res.status_code}): {res.text()}")

def get_model(url: str, headers = None)->Optional[str]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    return model_list[0]["id"] if model_list else None

def get_model_list(url: str, headers = None)->Optional[List[str]]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    if model_list and len(model_list) > 0:
        models = []
        for m in model_list:
            models.append(m["id"])
        return models
    else:
        return None

def main(args):
    print(f"Request: {args.url}")
    headers = {"User-Agent": "LLM API Client"}
    if args.api_key is not None and args.api_key != "":
        headers["Authorization"] = "Bearer " + args.api_key
    parsed_url = urlparse(args.url)
    if args.model_name is None:
        args.model_name = get_model(args.url[0:-len(parsed_url.path)] + "/v1/models")
    print(f"Model name: {args.model_name}")

    if "completions" in parsed_url.path:
        print("Call API: OpenAI completions")
        ctx = CompletionContext(prompt = args.prompt, model=args.model_name, max_tokens=args.output_len, stream=args.stream, ignore_eos=args.ignore_eos,)
        start_time = time.time()
        asyncio.run(async_request_openai_completions(ctx, args.url, args.api_key))
        duration = time.time() - start_time
        print(f"DONE in {duration:.3f} seconds")
        if ctx.error:
            print(f"ERROR: {ctx.error}")
        else:
            print(f"Generated: {ctx.generated}")
            print(f"E2E Latency: {ctx.e2e_latency:0.3f}, TTFT: {ctx.ttft:0.3f}")
    elif "embeddings" in parsed_url.path:
        print("Call API: embeddings")
        if args.prompt is None:
            raise ValueError("Invalid prompt to call embeddings API")
        result = asyncio.run(async_generate_embeddings(args.url, args.prompt, args.model_name, args.api_key))
        print(f"Embeddings response: {result}")
    elif "models" in parsed_url.path:
        print(f"Model: {get_model(args.url)}")
    elif "version" in parsed_url.path:
        print(f"Version: {get_version(args.url)}")
    elif "health" in parsed_url.path:
        print(f"Health check: {check_health(args.url)}")
    else:
        res = requests.get(args.url, headers=headers, timeout=10)
        print(f"Response: {res}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration about llm openapi.")
    parser.add_argument("--url", type=str, help="The host URL of the llm openapi server.", default="http://localhost:8000/v1/completions")
    parser.add_argument("--api-key", type=str, help="The secret key to connect server.")
    parser.add_argument("--prompt", type=str, help="The prompt for the completion.", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--model-name", type=str, help="The model name for completions.")
    parser.add_argument("--output-len", type=int, help="The maximum length of the output.", default=1024)
    parser.add_argument("--stream", action="store_true", help="Whether to stream the output or not.")
    parser.add_argument("--ignore-eos", action="store_true", help="Force to ouput the maximum length of the output, ignore eos when found.", default=None)

    args = parser.parse_args()
    main(args)
