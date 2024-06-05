import requests
import aiohttp
import argparse
import asyncio
import json
import numpy as np
import random
import requests
import subprocess
import time

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

STRICT_SEMAPHORE=False
PERCENTILES = [25, 50, 75, 90, 95, 99, 99.9, 99.99]

@dataclass
class Response:
    prompt: str = field(default="")
    generated: str = field(default="")
    prompt_len: int = field(default=0)
    output_len: int = field(default=0)
    latency: float = field(default=0.0)
    decode_latency: float = field(default=0.0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)

class AysncRequestSender:
    def __init__(self, backend: str, base_url: str, api_key: str, ep_models: str, ep_chat: str, ep_completion: str):
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.ep_models = ep_models
        self.ep_chat = ep_chat
        self.ep_completion = ep_completion
        self.responses = []
        self.headers = OrderedDict({"Content-Type": "application/json"})
        self.headers["Authorization"] = f"Bearer {self.api_key}"

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

    async def post_batch_requests_async(self,
        batch_size: int, requests: List[Tuple[str, int, int]],
        model: str, n: int, best_of: int, beam_search: bool, do_sample: bool, presence_penalty: float, frequency_penalty: float, repetition_penalty: float,
        temperature: float, top_p: float, top_k: int, stream: bool, eos_token_id: int,
    ):
        logger.info(f"post requests({len(requests)}), concurrency: {batch_size}, model: {model}, url: {self.base_url + self.ep_completion}")
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
            prompt, prompt_len, output_len = request
            task = asyncio.create_task(self._post_one_request(
                semaphore, model, prompt, prompt_len, output_len, n, best_of, beam_search, do_sample, presence_penalty, frequency_penalty, repetition_penalty, temperature, top_p, top_k, stream, eos_token_id
            ))
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

    async def _post_one_request(self, semaphore: asyncio.Semaphore, model: str, prompt: str, prompt_len: int, output_len: int,
        n: int, best_of: int, beam_search: bool, do_sample: bool, presence_penalty: float, frequency_penalty: float, repetition_penalty: float,
        temperature: float, top_p: float, top_k: int, stream: bool, eos_token_id: int,
    ):
        # prepare header, payload and params
        timeout = aiohttp.ClientTimeout(total=3600 * 3)
        retry = 1
        first_token_time = None
        ttft = None
        tpot = None
        output = ""
        if self.backend == "vllm":
            payload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "best_of": best_of,
                "use_beam_search": beam_search,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if do_sample else -1,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": stream,
            }
        elif self.backend == "novita":
            payload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": output_len,
                "stream": stream,
            }
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
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
                        async with session.post(self.base_url + self.ep_completion, headers = self.headers, json = payload) as res:
                            if res.status != 200:
                                logger.error(f"Failed to send request: {self.base_url + self.ep_completion}, status: {res.status}")
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
                                    if obj["choices"][0]["text"]:
                                        if ttft is None:
                                            first_token_time = time.perf_counter()
                                            ttft = first_token_time - request_start_time
                                        output += obj["choices"][0]["text"]
                        break
                    except json.decoder.JSONDecodeError as err:
                        logger.warning(f"Failed to load json string: {data}, error: {err}")
                        break
                    except Exception as err:
                        logger.warning(f"Failed to handle streaming chunk: {chunk}, error: {err}")
                        break
                    except aiohttp.ClientError as e:
                        if i < retry - 1:
                            logger.warning(f"Attempt {i + 1} failed: {e}, retrying")
                        else:
                            raise Exception(f"All {retry} attempts failed: {e}")

            # end of session
            if request_end_time == 0:
                request_end_time = time.perf_counter()
            request_latency = request_end_time - request_start_time
            decode_latency = request_end_time - first_token_time
            if output_len > 1:
                tpot = decode_latency / (output_len - 1)
            # logger.info(f"Generated text from server: {output}")
            self.responses.append(Response(prompt=prompt, generated=output, prompt_len=prompt_len, output_len=output_len, latency=request_latency, decode_latency=decode_latency, ttft=ttft, tpot=tpot))
            return True

    def save_response(self, file_path):
        with open(file_path, "a") as f:
            for res in self.responses:
                record = asdict(res)
                json.dump(record, f)
                f.write("\n")

    def dump_response_stats(self, tokenizer, stream, gpus, duration, log_file):
        if log_file is not None and log_file != "":
            self.save_response(log_file)

        num_responses = len(self.responses)
        if num_responses == 0:
            return
        total_generate_tokens = 0
        for response in self.responses:
            generated_tokens = tokenizer.encode(response.generated, add_special_tokens=False)
            generated_len = len(generated_tokens)
            total_generate_tokens += generated_len
            if generated_len == 0:
                logger.error("The generated tokens lenghth is 0")
                continue
            #logger.info(f"output len: {generated_len}, {response.output_len}")
            if abs(generated_len - response.output_len) > 10:
                logger.warning(f"expect generated {response.output_len} tokens, but got {len(generated_tokens)}.")
                response.tpot = round(response.decode_latency / generated_len, 2)

        total_prompt_tokens = np.sum([response.prompt_len for response in self.responses])
        total_tokens = total_prompt_tokens + total_generate_tokens
        tokens_per_sec = total_tokens / duration

        result_data = OrderedDict({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "num_requests": num_responses,
        })
        result_data["total_time"] = round(duration, 2)
        result_data["*tokens_per_second"] = int(total_tokens / duration)
        #result_data["*tokens_per_second_per_gpu"] = int(total_tokens / gpus / duration)
        result_data["prompt_tokens_per_second"] = int(total_prompt_tokens / duration)
        result_data["*output_tokens_per_second"] = int(total_generate_tokens / duration)
        #result_data["*output_tokens_per_second_per_gpu"] = int(total_generate_tokens / gpus / duration)
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

        print(f"============ Dump responses stats ==========================")
        print(f"TTFT(avg, p90): {result_data['*avg_TTFT']}, {result_data['TTFT_P90']}, out throughput: {1.0 / result_data['*avg_TPOT']: .1f}, total throughput(in, out): {result_data['prompt_tokens_per_second']}, {result_data['*output_tokens_per_second']}")
        print(f"Details:")
        max_key_length = max(len(str(k)) for k in new_data.keys())
        for key, value in new_data.items():
            print(f"{str(key):<{max_key_length}} | {str(value)}")

        if log_file is not None and log_file != "":
            with open(log_file, "a") as f:
                f.write("\n\n=============\n\n")
                json.dump(new_data, f)
                f.write("\n")


