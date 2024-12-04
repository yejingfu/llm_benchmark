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

DEF_REQUESTS = [
{
'temperature': 0, 'top_p': 1, 'presence_penalty': 0, 'frequency_penalty': 0, 'max_tokens': 1024, 'stop': ['<|eot_id|>', '<start_header_id|>', '<|end_header_id|>'],
'messages': [
{'content': "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{'LogisticsDashboard': {'type': 'object', 'properties': {'totalShipments': {'title': 'Total Shipments', 'type': 'integer'}, 'onTimeDeliveryRate': {'title': 'On Time Delivery Rate', 'type': 'number', 'minimum': 0, 'maximum': 100}, 'averageDeliveryTime': {'title': 'Average Delivery Time', 'type': 'number'}, 'pendingShipments': {'title': 'Pending Shipments', 'type': 'integer'}}, 'required': ['totalShipments', 'onTimeDeliveryRate', 'averageDeliveryTime', 'pendingShipments']}}\n</schema>\n", 'role': 'system'},
{'content': "I am currently working on a logistics dashboard for our air freight operations and I need to display some key performance indicators. Could you please construct a JSON object that includes the total number of shipments we've handled this quarter, the on-time delivery rate as a percentage, the average delivery time in hours, and the number of shipments that are still pending? Here are the details: We've handled a total of 523 shipments, our on-time delivery rate is at 96.5%, the average delivery time is approximately 18.2 hours, and there are 14 shipments still pending.", 'role': 'user'}
],
'guided_json': {'LogisticsDashboard': {'type': 'object', 'properties': {'totalShipments': {'title': 'Total Shipments', 'type': 'integer'}, 'onTimeDeliveryRate': {'title': 'On Time Delivery Rate', 'type': 'number', 'minimum': 0, 'maximum': 100}, 'averageDeliveryTime': {'title': 'Average Delivery Time', 'type': 'number'}, 'pendingShipments': {'title': 'Pending Shipments', 'type': 'integer'}}, 'required': ['totalShipments', 'onTimeDeliveryRate', 'averageDeliveryTime', 'pendingShipments']}}
}
]

async def send_one_request(index, req, args):
    url = args.endpoint + "/chat/completions"
    payload = req
    payload["model"] = args.model
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
                print(f"\n\nTestCase[{ctx.index}]\nE2E latency: {e2e_latency:.2f}, TTFT: {ttft:.2f}, sec/token: {(generate_latency/output_tokens):.3f}, tokens:{input_tokens}/{output_tokens}\n{generated}")

async def send_batch_requests(args):
    requests = DEF_REQUESTS[:]
    num = len(requests)
    t1 = time.perf_counter()
    print(f"Begin send {num} requests")
    tasks: List[asyncio.Task] = []
    for i in range(num):
        tasks.append(asyncio.create_task(send_one_request(i, requests[i], args)))
    await asyncio.gather(*tasks)
    t2 = time.perf_counter()
    print(f"End send {num} requests, time: {(t2 - t1):.2f}\n")


def main(args: argparse.Namespace):
    if not args.endpoint:
        raise ValueError("Invalid endpoint")
    if not args.model:
        raise ValueError("Invalid model name")
    asyncio.run(send_batch_requests(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON bad test cases"
    )
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--model", type=str, help="The model name")

    args = parser.parse_args()
    main(args)


