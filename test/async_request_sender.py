import requests
import aiohttp
import argparse
import asyncio
import json
import numpy as np
import random
import subprocess
import time
import copy

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional, Dict
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

PRINT_GENERATE_TEXT=False
STRICT_SEMAPHORE=False
PERCENTILES = [25, 50, 75, 90, 95, 99, 99.9, 99.99]

@dataclass
class RequestData:
    prompt: str = field(default="")
    prompt_len: int = field(default=0)
    ref_output: str = field(default="")
    max_tokens: int = field(default=0)

@dataclass
class Response:
    prompt: str = field(default="")
    generated: str = field(default="")
    prompt_len: int = field(default=0)
    ref_output: str = field(default="")
    output_len: int = field(default=0)
    latency: float = field(default=0.0)
    decode_latency: float = field(default=0.0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)

@dataclass
class InputParameter:
    model: str = field(default="")
    prompt: Optional[str] = field(default=None)
    messages: Optional[List[Dict[str, str]]] = field(default=None)
    n: Optional[int] = field(default=None)
    best_of: Optional[int] = field(default=None)
    use_beam_search: Optional[bool] = field(default=None)
    presence_penalty: Optional[float] = field(default=None)
    frequency_penalty: Optional[float] = field(default=None)
    repetition_penalty: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    max_tokens: Optional[int] = field(default=None)
    ignore_eos: Optional[bool] = field(default=None)
    stream: Optional[bool] = field(default=None)

    def to_dict(self):
        output = {}
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                output[k] = self.__dict__[k]
        return output


class AysncRequestSender:
    def __init__(self, backend: str, base_url: str, api_key: str, ep_models: str, ep_chat: str, ep_completion: str, sys_prompt: Optional[str]):
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.ep_models = ep_models
        self.ep_chat = ep_chat
        self.ep_completion = ep_completion
        self.responses = []
        self.headers = OrderedDict({"Content-Type": "application/json"})
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.sys_prompt = sys_prompt

    def check_health(self, retry: int) -> bool:
        ret = False
        logger.info(f"Check LLM service health: {self.base_url}/health")
        for i in range(retry):
            res = requests.get(self.base_url + "/health", headers = self.headers)
            if res.status_code == 200:
                ret = True
                break
            logger.info(f"try again to check health {i}")
            time.sleep(5)
        return ret

    def get_models(self)->Optional[str]:
        logger.info(f"Get models: {self.base_url + self.ep_models}")
        res = requests.get(self.base_url + self.ep_models, headers = self.headers)
        if res.status_code == 200:
            return res.text
        return None

    def get_response(self) -> List[Response]:
        return self.responses

    async def post_batch_requests_async(self, batch_size: int, requests: List[RequestData], parameters: InputParameter, chat_completions: bool):
        url = self.base_url + (self.ep_chat if chat_completions else self.ep_completion)
        logger.info(f"post requests({len(requests)}), concurrency: {batch_size}, model: {parameters.model}, url: {url}")
        self.responses = []
        tasks: List[asyncio.Task] = []
        progress_bar = async_tqdm(total=len(requests), desc="Processing Requests", smoothing=0.0)
        ## Create a single task to update semaphore
        semaphore = asyncio.Semaphore(batch_size)
        if STRICT_SEMAPHORE:
            for _ in range(batch_size - 1):
                await semaphore.acquire()
            tasks.append(asyncio.create_task(self._update_sem(semaphore, 1, 60, batch_size - 1, len(requests))))
        for request in requests:
            task = asyncio.create_task(self._post_one_request(semaphore, request, parameters, chat_completions))
            tasks.append(task)
            task.add_done_callback(lambda _: progress_bar.update())
        await asyncio.gather(*tasks)
        progress_bar.close()

    async def _update_sem(self, sem: asyncio.Semaphore, update_interval: int, ramp_up_period: int, batch_size: int, num_requests: int):
        n_parts = ramp_up_period // update_interval
        base = batch_size // n_parts
        remainder = batch_size % n_parts
        partitions = [base] * (n_parts - remainder) + [base + 1] * remainder
        for p in partitions:
            if len(self.responses) >= num_requests:
                break
            await asyncio.sleep(update_interval)
            for _ in range(p):
                sem.release()

    async def _post_one_request(self, semaphore: asyncio.Semaphore, request: RequestData, parameters: InputParameter, chat_completions: bool):
        # prepare header, payload and params
        timeout = aiohttp.ClientTimeout(total=3600 * 3)
        retry = 1
        first_token_time = None
        ttft = None
        tpot = None
        output = ""
        if chat_completions:
            parameters.messages = []
            if self.sys_prompt:
                parameters.messages.append({"role": "system", "content": self.sys_prompt})
            parameters.messages.append({"role": "user", "content": request.prompt})
        else:
            parameters.prompt = request.prompt
        parameters.max_tokens = request.max_tokens
        payload = parameters.to_dict()
        url = self.base_url + (self.ep_chat if chat_completions else self.ep_completion)
        # post now
        async with semaphore:
            request_start_time = time.perf_counter()
            request_end_time = 0
            first_token_time = 0
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for i in range(retry):
                    #logger.info(f"try: {i}")
                    try:
                        request_end_time = 0
                        output = ""
                        async with session.post(url, headers = self.headers, json = payload) as res:
                            if res.status != 200:
                                logger.error(f"Failed to send request: {url}, status: {res.status}, {res.text}")
                                return False
                            async for chunk_bytes in res.content:
                                chunk_bytes = chunk_bytes.strip()
                                if not chunk_bytes:
                                    continue
                                chunk = chunk_bytes.decode("utf-8")
                                if chunk.startswith("data: "):
                                    chunk = chunk[6:]
                                if chunk == "[DONE]":
                                    request_end_time = time.perf_counter()
                                else:
                                    obj = json.loads(chunk)
                                    content = None
                                    if "choices" in obj:
                                        choice0 = obj["choices"][0]
                                        if "text" in choice0: ## completions API
                                            content = choice0["text"]
                                        elif "delta" in choice0: ## chat-completions API
                                            #if "role" in choice0["delta"] and choice0["delta"]["role"] == "assistant" and "content" in choice0["delta"]:
                                            if "content" in choice0["delta"]:
                                                content = choice0["delta"]["content"]
                                    if ttft is None:
                                        first_token_time = time.perf_counter()
                                        ttft = first_token_time - request_start_time
                                    if content is not None:
                                        output += content

                        break
                    except json.decoder.JSONDecodeError as err:
                        logger.warning(f"Failed to load json string: {chunk}, error: {err}")
                        break
                    except Exception as err:
                        logger.warning(f"Failed to handle streaming chunk: {res.status}, error: {err}")
                        break
                    except aiohttp.ClientError as e:
                        if i < retry - 1:
                            logger.warning(f"Attempt {i + 1} failed: {e}, retrying")
                        else:
                            raise Exception(f"All {retry} attempts failed: {e}")

            # end of session
            if ttft is None or len(output) == 0:
                logger.warning("zero output, ignore it")
                return True
            if request_end_time == 0:
                request_end_time = time.perf_counter()
            request_latency = request_end_time - request_start_time
            decode_latency = request_end_time - first_token_time
            if PRINT_GENERATE_TEXT:
                logger.info(f"Generated text from server: {output}")
            self.responses.append(Response(prompt=request.prompt, generated=output, prompt_len=request.prompt_len, ref_output=request.ref_output, output_len=request.max_tokens, latency=request_latency, decode_latency=decode_latency, ttft=ttft, tpot=tpot))
            return True

    def save_response(self, file_path):
        with open(file_path, "a") as f:
            for res in self.responses:
                record = asdict(res)
                json.dump(record, f)
                f.write("\n")

    def dump_response_stats(self, tokenizer, stream, duration, print_dismatch, dump_res_details, prefix, log_file):
        if dump_res_details and log_file is not None and log_file != "":
            self.save_response(log_file)

        num_responses = len(self.responses)
        if num_responses == 0:
            return
        total_generate_tokens = 0
        ## calcuate the tpot for every response
        for response in self.responses:
            generated_tokens = tokenizer.encode(response.generated, add_special_tokens=False)
            generated_len = len(generated_tokens)
            total_generate_tokens += generated_len
            if generated_len == 0:
                logger.error("The generated tokens lenghth is 0")
                continue
            #logger.info(f"output len: {generated_len}, {response.output_len}")
            if abs(generated_len - response.output_len) > 10:
                if print_dismatch:
                    logger.warning(f"expect generated {response.output_len} tokens, but got {len(generated_tokens)}.")
            response.tpot = round(response.decode_latency / generated_len, 2) ## update the tpot by the correct generated token length

        total_prompt_tokens = np.sum([response.prompt_len for response in self.responses])
        total_tokens = total_prompt_tokens + total_generate_tokens
        tokens_per_sec = total_tokens / duration

        result_data = OrderedDict({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "num_requests": num_responses,
            "duration": duration,
        })
        result_data["total_time"] = round(duration, 2)
        result_data["*tokens_per_second"] = int(total_tokens / duration)
        result_data["prompt_tokens_per_second"] = int(total_prompt_tokens / duration)
        result_data["*output_tokens_per_second"] = int(total_generate_tokens / duration)
        result_data["requests_per_second"] = round(num_responses / duration, 2)

        # Compute the latency statistics.
        latencies = [response.latency for response in self.responses]
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        result_data["avg_latency"] = round(avg_latency, 3)
        result_data["min_latency"] = round(min_latency, 3)
        result_data["max_latency"] = round(max_latency, 3)
        latency_percentile = ", ".join(
            [
                f"P{k} = {v:.3f}"
                for k, v in zip(PERCENTILES, np.percentile(latencies, PERCENTILES))
            ]
        )
        result_data["percentile_latency"] = latency_percentile
        for percentile, v in zip(PERCENTILES, np.percentile(latencies, PERCENTILES)):
            result_data[f"latency_P{percentile}"] = round(v, 3)

        avg_per_token_latency = np.mean(
            [response.latency / (response.prompt_len + response.output_len) for response in self.responses]
        )
        result_data["avg_latency_per_prompt_token"] = round(avg_per_token_latency, 3)
        avg_per_output_token_latency = np.mean(
            [response.latency / response.output_len for response in self.responses]
        )
        result_data["avg_latency_per_output_token"] = round(avg_per_output_token_latency, 3)

        if stream:
            ttft = [response.ttft for response in self.responses]
            ttft_percentile = ", ".join(
                [f"P{k} = {v:.3f}" for k, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES))]
            )
            result_data["ttft_percentile"] = ttft_percentile
            result_data["*avg_TTFT"] = round(np.mean(ttft), 3)
            result_data["avg_prompt_tokens_per_secend"] = round(total_prompt_tokens / num_responses / np.mean(ttft), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES)):
                result_data[f"TTFT_P{percentile}"] = round(v, 3)
            tpot = [response.tpot or np.nan for response in self.responses]
            tpot_percentile = ", ".join(
                [f"P{k} = {v:.3f}" for k, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES))]
            )
            result_data["tpot_percentile"] = tpot_percentile
            result_data["*avg_TPOT"] = round(np.mean(tpot), 3)
            result_data["avg_output_tokens_per_second"] = round(1 / np.mean(tpot), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES)):
                result_data[f"TPOT_P{percentile}"] = round(v, 3)

        new_data = OrderedDict()
        #for k, v in result_data.items():
        #    k = k.lower()
        #    new_data[k] = v
        for k, v in result_data.items():
            if k not in new_data:
                new_data[k] = v

        print(f"============ Dump responses stats: {prefix} ==========================")
        print(f"[Latency] E2E(avg, p90): {result_data['avg_latency']:0.2f}, {result_data['latency_P90']:0.2f}")
        print(f"[Latency] TTFT(avg, p90): {result_data['*avg_TTFT']:0.2f}, {result_data['TTFT_P90']:0.2f}; TPOT(avg, p90): {result_data['*avg_TPOT']: .3f}, {result_data['TPOT_P90']: .3f}")
        print(f"[Throughput] (input, output): {result_data['prompt_tokens_per_second']}, {result_data['*output_tokens_per_second']}")
        print(f"Details:")
        max_key_length = max(len(str(k)) for k in new_data.keys())
        for key, value in new_data.items():
            print(f"{str(key):<{max_key_length}} | {str(value)}")

        if log_file is not None and log_file != "":
            with open(log_file, "a") as f:
                f.write(f"============ Dump responses stats: {prefix} ==========================\n")
                f.write(f"[Latency] E2E(avg, p90): {result_data['avg_latency']:0.2f}, {result_data['latency_P90']:0.2f}\n")
                f.write(f"[Latency] TTFT(avg, p90): {result_data['*avg_TTFT']:0.2f}, {result_data['TTFT_P90']:0.2f}; TPOT(avg, p90): {result_data['*avg_TPOT']: .3f}, {result_data['TPOT_P90']: .3f}\n")
                f.write(f"[Throughput] (input, output): {result_data['prompt_tokens_per_second']}, {result_data['*output_tokens_per_second']}\n")
                f.write(f"Details:\n")
                max_key_length = max(len(str(k)) for k in new_data.keys())
                for key, value in new_data.items():
                    f.write(f"{str(key):<{max_key_length}} | {str(value)}\n")
                f.write("\n\n")

                #f.write("\n\n=============\n\n")
                #json.dump(new_data, f)
                #f.write("\n")


