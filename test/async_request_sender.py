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

PERCENTILES = [25, 50, 75, 90, 95, 99, 99.9, 99.99]

@dataclass
class Response:
    prompt: str = field(default="")
    generated: str = field(default="")
    prompt_len: int = field(default=0)
    output_len: int = field(default=0)
    latency: float = field(default=0.0)
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
        for i in range(retry):
            res = requests.get(self.base_url + "/health", headers = self.headers)
            if res.status_code == 200:
                ret = True
                break
            logger.info("try again to check health {i}")
        return ret

    def get_models(self)->Optional[str]:
        res = requests.get(self.base_url + self.ep_models, headers = self.headers)
        if res.status_code == 200:
            return res.text
        return None

    def get_response(self) -> List[Response]:
        return self.response

    async def post_batch_requests_async(self,
        batch_size: int, requests: List[Tuple[str, int, int]],
        model: str, n: int, best_of: int, beam_search: bool, do_sample: bool, presence_penalty: float, frequency_penalty: float, repetition_penalty: float,
        temperature: float, top_p: float, top_k: int, stream: bool, eos_token_id: int,
    ):
        self.response = []
        semaphore = asyncio.Semaphore(batch_size)
        for _ in range(batch_size - 1):
            await semaphore.acquire()
        ## Create a single task to update semaphore
        tasks: List[asyncio.Task] = [asyncio.create_task(self._update_sem(semaphore, 1, 60, batch_size - 1))]
        progress_bar = async_tqdm(total=len(requests), desc="Processing Requests", smoothing=0.0)
        for request in requests:
            prompt, prompt_len, output_len = request
            task = asyncio.create_task(self._post_one_request(
                semaphore, model, prompt, prompt_len, output_len, n, best_of, beam_search, do_sample, presence_penalty, frequency_penalty, repetition_penalty, temperature, top_p, top_k, stream, eos_token_id
            ))
            tasks.append(task)
            task.add_done_callback(lambda _: progress_bar.update())
        await asyncio.gather(*tasks)
        progress_bar.close()

    async def _update_sem(self, sem: asyncio.Semaphore, update_interval: int, ramp_up_period: int, batch_size: int):
        n_parts = ramp_up_period // update_interval
        base = batch_size // n_parts
        remainder = batch_size % n_parts
        partitions = [base] * (n_parts - remainder) + [base + 1] * remainder
        for p in partitions:
            await asyncio.sleep(update_interval)
            for _ in range(p):
                sem.release()

    async def _post_one_request(self, semaphore: asyncio.Semaphore, model: str, prompt: str, prompt_len: int, output_len: int,
        n: int, best_of: int, beam_search: bool, do_sample: bool, presence_penalty: float, frequency_penalty: float, repetition_penalty: float,
        temperature: float, top_p: float, top_k: int, stream: bool, eos_token_id: int,
    ):
        # prepare header, payload and params
        timeout = aiohttp.ClientTimeout(total=3600 * 3)
        retry = 10
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
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for i in range(retry):
                    try:
                        async with session.post(self.base_url + self.ep_completion, headers = self.headers, json = payload) as res:
                            chunks = []
                            async for chunk, _ in res.content.iter_chunks():
                                chunks.append(chunk)
                                if ttft is None:
                                    # got first token
                                    first_token_time = time.perf_counter()
                                    ttft = first_token_time - request_start_time
                        # returned from post
                        if output_len > 1:
                            tpot = (time.perf_counter() - first_token_time) / (output_len - 1)
                        if stream:
                            outputs = []
                            for chunk in chunks:
                                try:
                                    data = chunk.rstrip(b"\x00").lstrip(b"data:").rstrip(b"\n\n").strip().decode("utf-8")
                                    obj = json.loads(data)
                                    outputs.append(obj["choices"][0]["text"])
                                except json.decoder.JSONDecodeError as err:
                                    logger.warning(f"Failed to load json string: {data}, error: {err}")
                                    continue
                                except Exception as err:
                                    logger.warning(f"Failed to handle streaming chunk: {chunk}, error: {err}")
                                    continue
                            output = "".join(outputs)
                        else:
                            output = b"".join(chunks).decode("utf-8")
                            # output = json.loads(output)
                        break
                    except aiohttp.ClientError as e:
                        if i < retry - 1:
                            logger.warning(f"Attempt {i + 1} failed: {e}, retrying")
                        else:
                            raise Exception(f"All {retry} attempts failed: {e}")

            # end of session
            request_end_time = time.perf_counter()
            request_latency = request_end_time - request_start_time
            self.responses.append(Response(prompt=prompt, generated=output, prompt_len=prompt_len, output_len=output_len, latency=request_latency, ttft=ttft, tpot=tpot))

    def save_response(self, file_path):
        record = asdict(response)
        with open(file_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def dump_response_stats(self, tokenizer, stream, gpus, duration, log_file):
        num_responses = len(self.responses)
        if num_response == 0:
            return
        generate_tokens = 0
        for response in self.responses:
            generated_tokens = tokenizer.encode(response.generated, add_special_tokens=False)
            generated_len = len(generated_tokens)
            generate_tokens += generated_len
            if abs(generated_len - response.output_len) > 20:
                logger.warning(f"expect generated {response.output_len} tokens, but got {len(generated_tokens)}.")

        prompt_tokens = np.sum([response.prompt_len for response in self.responses])
        total_tokens = prompt_tokens + generate_tokens
        tokens_per_sec = total_tokens / duration

        result_data = OrderedDict({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "num_requests": num_responses,
        })
        result_data["total_time"] = round(duration, 2)
        result_data["tokens_per_second"] = int(total_tokens / duration)
        result_data["tokens_per_second_per_gpu"] = int(total_tokens / gpus / duration)
        result_data["prompt_tokens_per_second"] = int(prompt_tokens / duration)
        result_data["output_tokens_per_second"] = int(generate_tokens / duration)
        result_data["output_tokens_per_second_per_gpu"] = int(generate_tokens / gpus / duration)
        result_data["requests_per_second"] = round(num_responses / duration, 2)

        logger.info(f"============ Dump responses stats ==========================")
        max_key_length = max(len(str(k)) for k in result_data.keys())
        for key, value in result_data.items():
            print(f"{str(key):<{max_key_length}} | {str(value)}")

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

        if tream:
            ttft = [response.ttft for response in self.responses]
            ttft_percentile = ", ".join(
                [f"P{k} = {v:.3f}" for k, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES))]
            )
            result_data["ttft_percentile"] = ttft_percentile
            logger.info(f"{phase} TTFT: avg = {np.mean(ttft):.3f}({(total_tokens - generate_tokens) / num_response / np.mean(ttft):.3f}), {ttft_percentile}")
            result_data["avg_TTFT"] = round(np.mean(ttft), 3)
            result_data["avg_prompt_tokens_per_secend"] = round((total_tokens - generate_tokens) / num_responses / np.mean(ttft), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(ttft, PERCENTILES)):
                result_data[f"TTFT_P{percentile}"] = round(v, 3)
            tpot = [response.tpot or np.nan for response in self.responses]
            tpot_percentile = ", ".join(
                [f"P{k} = {v:.3f}" for k, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES))]
            )
            result_data["tpot_percentile"] = tpot_percentile
            logger.info(f"{phase} TPOT: avg = {np.mean(tpot):.3f} ({1 / np.mean(tpot):.3f}), {tpot_percentile}")
            result_data["avg_TPOT"] = round(np.mean(tpot), 3)
            result_data["avg_output_tokens_per_second"] = round(1 / np.mean(tpot), 3)
            for percentile, v in zip(PERCENTILES, np.percentile(tpot, PERCENTILES)):
                result_data[f"TPOT_P{percentile}"] = round(v, 3)

        new_data = OrderedDict()
        for k, v in result_data.items():
            k = k.lower()
            new_data[k] = v
        for k, v in result_data.items():
            if k not in new_data:
                new_data[k] = v

        logger.info(f"============ Dump responses stats ==========================")
        max_key_length = max(len(str(k)) for k in new_data.keys())
        for key, value in new_data.items():
            print(f"{str(key):<{max_key_length}} | {str(value)}")

        if log_file is not None:
            with open(log_file, "a") as f:
                json.dump(new_data, f)
                f.write("\n")


