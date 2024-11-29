import argparse
import time
import os
import aiohttp
import asyncio
import json
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from langchain.evaluation import JsonValidityEvaluator
# calculate strings similarity (import difflib)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


### load dataset from huggingface: https://huggingface.co/datasets/NousResearch/json-mode-eval

@dataclass
class Context:
    index: int = field(default=0)
    prompts: object = field(default=None)
    schema: object = field(default=None)
    verify: object = field(default=None)
    output: object = field(default=None)

def read_json_from_parquet(file: str):
    df = pd.read_parquet(file)
    dataset = df.to_dict("records")
    all_ctx = []
    num = 0
    for record in dataset:
        if "prompt" in record and "completion" in record and "schema" in record:
            ctx = Context(index=num)
            ctx.prompts = record["prompt"].tolist()
            ctx.verify = record["completion"]
            ctx.schema = json.loads(record["schema"])
            #ctx.schema = {}
            #for k, v in record["schema"].items():
            #    ctx.schema[k] = v
            all_ctx.append(ctx)
            num += 1
    return all_ctx

def get_chat_payload(ctx, args):
    obj = {
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 1024,
        #"repetition_penalty": 1,
        "stop": ["<|eot_id|>", "<start_header_id|>", "<|end_header_id|>"],
        "model": args.model,
        "messages": ctx.prompts,
    }
    if not args.no_json:
        obj["guided_json"] = ctx.schema
    return obj

async def send_one_request(index, ctx, args):
    url = args.endpoint + "/chat/completions"
    payload = get_chat_payload(ctx, args)
    #print(f"==== payload: {payload}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        request_start_time = time.perf_counter()
        async with session.post(url, headers=OrderedDict({"Content-Type": "application/json"}), json=payload) as res:
            if res.status != 200:
                text = await res.text()
                print(f"ERROR: {res.status}--{res.reason}: {text}")
                print(f"request: {payload}")
            else:
                generated = ""
                e2e_latency = 0
                first_token_ts = 0
                input_tokens = 0
                output_tokens = 0
                async for chunk_bytes in res.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    try:
                        chunk = chunk_bytes.decode("utf-8")
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        #print(f"==== chunk: ++{chunk}++")
                        if chunk == ": OPENROUTER PROCESSING":
                            continue
                        if chunk == "[DONE]":
                            e2e_latency = time.perf_counter() - request_start_time
                        else:
                            if first_token_ts == 0:
                                first_token_ts = time.perf_counter()
                            obj = json.loads(chunk)
                            #print(f"=== output json: {obj}")
                            content = None
                            if "choices" in obj:
                                choice0 = obj["choices"][0]
                                if "text" in choice0:
                                    content = choice0["text"]
                                elif "delta" in choice0:
                                    if "content" in choice0["delta"]:
                                        content = choice0["delta"]["content"]
                                elif "message" in choice0:
                                    if "content" in choice0["message"]:
                                        content = choice0["message"]["content"]
                                output_tokens += 1
                            if "usage" in obj:
                                input_tokens = obj["usage"]["prompt_tokens"]
                                output_tokens = obj["usage"]["completion_tokens"]
                            if content is not None:
                                generated += content
                    except json.decoder.JSONDecodeError as err:
                        print(f"JSON DECODE ERROR: {chunk}, {err}")
                    except Exception as err:
                        print(f"Failed to handle streaming chunk: {res.status}, error: {err}")
                if e2e_latency == 0:
                    ## non-stream
                    e2e_latency = time.perf_counter() - request_start_time
                    generate_latency = e2e_latency
                    ttft = 0.0
                else:
                    ## stream
                    assert first_token_ts > 0
                    generate_latency = time.perf_counter() - first_token_ts
                    ttft = first_token_ts - request_start_time
                ctx.output = generated
                print(f"\n\nTestCase[{ctx.index}]\nE2E latency: {e2e_latency:.2f}, TTFT: {ttft:.2f}, sec/token: {(generate_latency/output_tokens):.3f}, tokens:{input_tokens}/{output_tokens}\n{generated}")
                if not args.no_json:
                    print(f"{ctx.verify} (reference)")

async def send_batch_requests(contexts, args):
    num = len(contexts)
    t1 = time.perf_counter()
    print(f"Begin send {num} requests")
    tasks: List[asyncio.Task] = []
    for i in range(num):
        tasks.append(asyncio.create_task(send_one_request(i, contexts[i], args)))
    await asyncio.gather(*tasks)
    t2 = time.perf_counter()
    print(f"End send {num} requests, time: {(t2 - t1):.2f}\n")


def main(args: argparse.Namespace):
    if not args.endpoint:
        raise ValueError("Invalid endpoint")
    if not args.model:
        raise ValueError("Invalid model name")
    if not args.dataset or not os.path.isfile(args.dataset):
        raise ValueError(f"Invalid dataset path: {args.dataset}")
    ctxes = read_json_from_parquet(args.dataset)
    #ctxes = ctxes[0:10]
    total = len(ctxes)
    print(f"Read records: {len(ctxes)}")
    i = 0
    while i < total:
        j = i + args.parallel
        if j > total:
            j = total
        asyncio.run(send_batch_requests(ctxes[i:j], args))
        i += args.parallel
    print("===== Summary ======")
    vectorizer = CountVectorizer()
    evaluator = JsonValidityEvaluator()
    json_scores = 0
    sum_similarity = 0
    num = 0
    for ctx in ctxes:
        if ctx.output:
            num += 1
            result = evaluator.evaluate_strings(prediction=ctx.output)
            vectors = vectorizer.fit_transform([ctx.output, ctx.verify])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            print(f"verify[{ctx.index}]: {result}, {similarity}")
            json_scores += result["score"]
            sum_similarity += similarity
    if num > 0:
        print(f"Pass@{num}: {(json_scores/num):.2f}, similarity: {sum_similarity/num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON performance evaluation"
    )
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--model", type=str, help="The model name")
    parser.add_argument("--dataset", type=str, help="The parquet file containing the JSON test cases")
    parser.add_argument("--parallel", type=int, default=1, help="The num of requests sent in parallel")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output if set")

    args = parser.parse_args()
    main(args)

