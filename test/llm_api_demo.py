import argparse
import asyncio
import json
import os
import sys
import traceback
import time
import warnings
import aiohttp
import requests
from loguru import logger
from tqdm.asyncio import tqdm
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

VLLM_ENDPOINTS = {
    "version": "/version",
    "health": "/health",
    "models": "/v1/models",
    "completions": "/v1/completions",
    "chat-completions": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "compatible-embeddings": "/compatible-mode/v1/embeddings"
}

@dataclass
class RequestInput:
    prompt: str
    api_url: str
    prompt_len: int = 0
    output_len: int = 0
    model: str = ""
    stream: bool = True
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = None
    ignore_eos: Optional[bool] = None

@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""

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

async def async_request_openai_completions(
    action: str,
    request_input: RequestInput,
    api_key: Optional[str] = None,
    pbar: Optional[tqdm] = None,
) -> RequestOutput:
    api_url = request_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_input.use_beam_search
        if os.path.isfile(request_input.prompt) and os.path.exists(request_input.prompt):
            with open(request_input.prompt, "r") as f:
                request_input.prompt = f.read()
                logger.info(f"Load prompt from file: {request_input.prompt}")
        payload = {
            "model": request_input.model,
            #"temperature": 0.8,
            "max_tokens": request_input.output_len,
            "stream": request_input.stream,
        }
        if request_input.best_of is not None:
            payload["best_of"] = request_input.best_of
        if request_input.ignore_eos is not None:
            payload["ignore_eos"] = request_input.ignore_eos
        if action == "completions":
            payload["prompt"] = request_input.prompt
        elif action == "chat-completions":
            payload["messages"] = [{"role": "user", "content": request_input.prompt}]

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

        output = RequestOutput()
        output.prompt_len = request_input.prompt_len

        generated_text = ""
        ttft = 0.0
        latency = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        print(f"URL:{api_url}, payload: {payload}")
        try:
            iter = 0
            async with session.post(url=api_url, json=payload, headers=headers) as response:
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
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)
                            if request_input.stream:
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
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                # Decoding phase
                                elif data.get("usage", None) is None:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    if not request_input.stream:
                        print(output.generated_text)
                else:
                    output.success = False
                    details = await response.text()
                    output.error = f"status code {response.status}({response.reason}).\n\t Details: {details}."
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

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
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def get_version(url: str):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.text
        else:
            print(f"get_version: {res}")
    except e:
        print(f"Exception is raised: {e}")
    return ""

def exec_get_method(args):
    try:
        headers = {"User-Agent": "LLM API Client"}
        if args.api_key is not None and args.api_key != "":
            headers["Authorization"] = "Bearer " + args.api_key
        res = requests.get(args.url + args.path, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.text
        else:
            print(f"Response: {res}")
    except e:
        print(f"Exception is raised: {e}")
    return ""


def list_models(url: str):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.text
        else:
            print(f"list_models: {res}")
    except e:
        print(f"Exception is raised: {e}")
    return ""

def main(args):
    if args.backend != "vllm":
        print("Only support vllm")
        return
    action = None
    for k, v in VLLM_ENDPOINTS.items():
        if args.path == v:
            action = k
            break
    #if not action:
    #    print(f"Invalid path: {args.path}")
    #    return
    url = args.url + args.path
    print(f"Request: {url}")

    if action == "health":
        if check_health(url):
            print("Health check is passed")
        else:
            print("Health check is failed")
    elif action == "version":
        ver_str = get_version(url)
        if ver_str:
            ver = json.loads(ver_str)
            print(f"Version: {ver['version']}")
        else:
            print(f"unknown version: {ver_str}")
    elif action == "models":
        models_str = list_models(url)
        if models_str:
            models = json.loads(models_str)
            index = 1
            for m in models['data']:
                print(f"Model[{index}]: {m['id']}, {m['object']}")
                index += 1
        else:
            print(f"unknown models: {models_str}")
    elif action == "completions" or action == "chat-completions":
        input = RequestInput(
            api_url = url,
            prompt = args.prompt,
            model = args.model_name,
            output_len = args.output_len,
            stream = args.stream,
            ignore_eos = args.ignore_eos,
            best_of = args.best_of,
        )
        print(input)
        start_time = time.time()
        result = None
        if args.repeat > 1:
            async def _run_requests_in_batch(num):
                tasks: List[asyncio.Task] = []
                for _ in range(num):
                    tasks.append(asyncio.create_task(async_request_openai_completions(action, input, api_key=args.api_key)))
                await asyncio.gather(*tasks)
            asyncio.run(_run_requests_in_batch(args.repeat))
        else:
            result = asyncio.run(async_request_openai_completions(action, input, api_key=args.api_key))
        end_time = time.time()
        #print(f"Time elapsed: {datetime.fromtimestamp(end_time - start_time).strftime('%H:%M:%S.%f')}")
        print(f"\n Time elapsed: {end_time - start_time} seconds.")
        if result is not None:
            if result.success:
                print(f"Final generated text: {result.generated_text}")
            else:
                print(f"Error: {result.error}")
    elif action == "embeddings" or action == "compatible-embeddings":
        if args.url is None or args.model_name is None or args.prompt is None:
            logger.error("Invalid embeddings paramters, required arguments: url, model-name, prompt")
            return
        result = asyncio.run(async_generate_embeddings(url, args.prompt, args.model_name, args.api_key))
        logger.info(f"Embbedings result: {result}")
    else:
        print(f"execute GET method: {url}")
        ret = exec_get_method(args)
        print(ret)
        return
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration about llm openapi.")
    parser.add_argument("--url", type=str, help="The host URL of the llm openapi server.", default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, help="The secret key to connect server.", default="")
    parser.add_argument("--backend", type=str, help="The backend of the llm openapi server.", default="vllm", choices=["vllm", "trtllm"])
    parser.add_argument("--path", type=str, help=f"The URL path of the llm openapi server. Its values can be: {VLLM_ENDPOINTS}", default="/v1/completions")
    parser.add_argument("--prompt", type=str, help="The prompt for the completion.", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--model-name", type=str, help="The model name for completions.", default="llama3-8b")
    parser.add_argument("--output-len", type=int, help="The maximum length of the output.", default=1024)
    parser.add_argument("--ignore-eos", action="store_true", help="Force to ouput the maximum length of the output, ignore eos when found.", default=None)
    parser.add_argument("--best-of", type=int, help="The number of best candicates in the sampling phase")
    parser.add_argument("--stream", action="store_true", help="Whether to stream the output or not.")
    parser.add_argument("--repeat", type=int, help="The number to repeat the same requests.", default=1)

    args = parser.parse_args()
    main(args)
