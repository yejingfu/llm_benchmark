#!/usr/bin/env python

"""
The fast version to execute LLM benchmark.
reference: https://github.com/fixie-ai/ai-benchmarks
"""
import argparse
import asyncio
import os
import json
import time
import aiohttp
import numpy as np
from datetime import datetime
from loguru import logger
from typing import List
from dataclasses import dataclass, field, asdict

import llm_request

PROMPT_DEF = "Please instroduce Shanghai China to a foreign tourist in Englisth."
PROMPT_SUMARIZE_1000 = """
Summarize the document below into a single informative sentence, and based on the below content write a detailed instroduction:
---
Foundation models are general models of language, vision, speech, and/or other modalities that are designed to support a large variety of AI tasks. They form the basis of many modern AI systems.
The development of modern foundation models consists of two main stages: (1) a pre-training stage in which the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning).
In this paper, we present a new set of foundation models for language, called Llama 3. The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens. Each member of the herd is listed in Table 1. All the results presented in this paper are for the Llama 3.1 models, which we will refer to as Llama 3 throughout for brevity.
We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity. We seek to optimize for these three levers in our development process:
Data. Compared to prior versions of Llama (Touvron et al., 2023a,b), we improved both the quantity and quality of the data we use for pre-training and post-training. These improvements include the development of more careful pre-processing and curation pipelines for pre-training data and the development of more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.
Scale. We train a model at far larger scale than previous Llama models: our flagship language model was pre-trained using 3.8 × 1025 FLOPs, almost 50× more than the largest version of Llama 2. Specifically, we pre-trained a flagship model with 405B trainable parameters on 15.6T text tokens. As expected per. scaling laws for foundation models, our flagship model outperforms smaller models trained using the same procedure. While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget. We use the flagship model to further improve the quality of those smaller models during post-training.
Managing complexity. We make design choices that seek to maximize our ability to scale the model development process. For example, we opt for a standard dense Transformer model architecture (Vaswani et al., 2017) with minor adaptations, rather than for a mixture-of-experts model (Shazeer et al., 2017) to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO; Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al., 2022; Schulman et al., 2017) that tend to be less stable and harder to scale.
The result of our work is Llama 3: a herd of three multilingual language models with 8B, 70B, and 405B parameters. We evaluate the performance of Llama 3 on a plethora of benchmark datasets that span a wide range of language understanding tasks. In addition, we perform extensive human evaluations that compare Llama 3 with competing models. An overview of the performance of the flagship Llama 3 model on key benchmarks is presented in Table 2. Our experimental evaluation suggests that our flagship model performs on par with leading language models such as GPT-4 (OpenAI, 2023a) across a variety of tasks, and is close to matching the state-of-the-art. Our smaller models are best-in-class, outperforming alternative models with similar numbers of parameters (Bai et al., 2023; Jiang et al., 2023). Llama 3 also delivers a much better balance between helpfulness and harmlessness than its predecessor (Touvron et al., 2023b). We present a detailed analysis of the safety of Llama 3 in Section 5.4.
We are publicly releasing all three Llama 3 models under an updated version of the Llama 3 Community License; see https://llama.meta.com.
"""

DEFAULT_PROMPTS = [PROMPT_SUMARIZE_1000]
DEFAULT_MAX_TOKENS = 1000
DEFAULT_NUM_REQUESTS = 200
DEFAULT_PARALLEL = [10, 20, 30]

class TestElement:
    def __init__(self, name: str, model: str, endpoint: str, api_key: str):
        self.provider = name
        self.model_name = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.duration = []
        self.TTFT = []
        self.TPOT = []

    @property
    def display_name(self):
        return self.provider + "/" + self.model_name

@dataclass
class LlmInputArgs:
    prompt: str = field(default="")
    model: str = field(default="")
    strict: bool = field(default=False)
    detail: str = field(default=None)
    temperature: float = field(default=0)
    max_tokens: int = field(default=100)
    api_key: str = field(default=None)
    base_url: str = field(default="")
    peft: str = field(default=None)


class LlmTraceConfig(aiohttp.TraceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.on_request_start.append(self._on_request_start_func)
        self.on_connection_create_end.append(self._on_connection_create_end_func)
        self.on_connection_reuseconn.append(self._on_connection_reuseconn_func)
        self.on_request_headers_sent.append(self._on_request_headers_sent_func)
        self.on_request_chunk_sent.append(self._on_request_chunk_sent_func)

    async def _on_request_start_func(self, session, ctx, params):
        ctx.url = params.url
        ctx.start_time = time.time()

    async def _on_connection_create_end_func(self, session, ctx, params):
        self._trace(ctx, "created connection")

    async def _on_connection_reuseconn_func(self, session, ctx, params):
        self._trace(ctx, "reused connection")

    async def _on_request_headers_sent_func(self, session, ctx, params):
        self._trace(ctx, "sent headers")

    async def _on_request_chunk_sent_func(self, session, ctx, params):
        #self._trace(ctx, "sent chunk")
        pass

    def _trace(self, ctx, action):
        delta = time.time() - ctx.start_time
        print(f"[{delta:.3f}] {ctx.url.host}: {action}")

async def run_tests(element: TestElement, num_requests: int, dump: str):
    # construct context
    connector = aiohttp.TCPConnector(force_close=False)
    trace_configs = [LlmTraceConfig()]
    timeout = aiohttp.ClientTimeout(total=3600*30)
    args = LlmInputArgs(model=element.model_name, max_tokens=DEFAULT_MAX_TOKENS, api_key=element.api_key, base_url=element.endpoint)
    func = llm_request.openai_chat
    dump_str = ""
    async with aiohttp.ClientSession(timeout=timeout, trace_configs=trace_configs, connector=connector) as session:
        contexts = []
        for index in range(num_requests):
            args.prompt = DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]
            contexts.append(llm_request.ApiContext(session, index, element.display_name, func, args, args.prompt, [], []))
        def _on_token(ctx: llm_request.ApiContext, token: str):
            if token:
                print(token, end="", flush=True)
        for parallel in DEFAULT_PARALLEL:
            logger.info(f"Start requesting in parallel: {parallel}, total {num_requests}, model {element.display_name}")
            time_start = time.perf_counter()
            for i in range(0, num_requests, parallel):
                tasks = [asyncio.create_task(ctx.run()) for ctx in contexts[i : i + parallel]]
                await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - time_start
            # dump
            dump_str += f"\n==== {element.display_name}, parallel {parallel} / {num_requests} @ {datetime.now()}=====\nDuration: {elapsed*1000:.0f} ms\n"
            metrics = [ctx.metrics for ctx in contexts if not ctx.metrics.error]
            metrics_error = [ctx.metrics for ctx in contexts if ctx.metrics.error]
            for merr in metrics_error:
                dump_str += f"[error] {merr.model}, {merr.error}\n"
            ttft = []
            tps = []
            output_tokens = []
            if len(metrics) > 0:
                for m in metrics:
                    ttft.append(m.ttft)
                    tps.append(m.tps)
                    output_tokens.append(m.output_tokens)
                p_ttft = np.percentile(ttft, [90, 95])
                p_tps = np.percentile(tps, [90, 95])
                dump_str += f"TTFT(min,avg,p95,max): {np.min(ttft)*1000:.0f},{np.mean(ttft)*1000:.0f},{p_ttft[1]*1000:.0f},{np.max(ttft)*1000:.0f}\n"
                dump_str += f"TPS(min,avg,p95,max): {np.min(tps):.0f},{np.mean(tps):.0f},{p_tps[1]:.0f},{np.max(tps):.0f}\n"
                dump_str += f"sysTPS: {np.sum(output_tokens)}/{elapsed:.3f} = {np.sum(output_tokens)/elapsed:.0f}\n"
                dump_str += f"[request 0] queue_time: {metrics[0].provider_queue_time}, input_time: {metrics[0].provider_input_time}, completion_time: {metrics[0].provider_output_time}, total_time:{metrics[0].provider_total_time}\n"
                dump_str += f"[request 0] input_tokens: {metrics[0].input_tokens}, ouput_tokens: {metrics[0].output_tokens}, total time: {metrics[0].total_time:.3f}\n"
            if dump:
                with open(dump, "a") as f:
                    f.write(dump_str)
            logger.info(f"End requesting\n{dump_str}\nSaved into {dump}\n")


def main(args: argparse.Namespace):
    json_data = None
    providers_file = args.providers
    if providers_file is None:
        providers_file = os.path.dirname(os.path.abspath(__file__)) + "/llm_providers.json"
    if os.path.isfile(providers_file):
        with open(providers_file, "r") as f:
            json_data = json.load(f)
    if json_data is None:
        raise RuntimeError(f"Failed load provider json file from {providers_file}")
    elements: List[TestElement] = []
    if "providers" in json_data and len(json_data["providers"]) > 0:
        providers = json_data["providers"]
        for p in providers:
            if "enable" in p and p["enable"] == False:
                continue
            name = p["name"]
            model_names = p["model-names"]
            ep = p["endpoint"]
            ak = p["api_key"]
            for n in model_names:
                element = TestElement(name, n, ep, ak)
                if args.filter is None or args.filter.lower() in element.display_name.lower():
                    elements.append(element)
    if len(elements) == 0:
        raise RuntimeError("No valid provider found")
    time_start = time.perf_counter()
    tasks = []
    for e in elements:
        if e.model_name is not None and e.endpoint is not None:
            asyncio.run(run_tests(e, DEFAULT_NUM_REQUESTS, args.dump_file))
    elapsed = time.perf_counter() - time_start
    logger.info(f"DONE in {elapsed:.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM benchmark fast testing tool.")
    parser.add_argument("--providers", type=str, help="The json file to contain provider list. If not set, read from llm_providers.json")
    parser.add_argument("--filter", type=str, help="The key words to filter out providers")
    parser.add_argument("--dump-file", type=str, help="Append the test results to the file")
    args = parser.parse_args()
    main(args)

