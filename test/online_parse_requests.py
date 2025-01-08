import argparse
import asyncio
import json
import csv
import os
import sys
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict, Any
from transformers import AutoTokenizer
from loguru import logger

@dataclass
class RawRequest:
    model: str = field(default="")
    prompt: Optional[str] = field(default=None)
    messages: Optional[List[Dict[str, str]]] = field(default=None)
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
    # statistics
    prompt_len: Optional[int] = field(default=None)

def decode_unicode_strings(text):
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

def create_request(obj):
    req = RawRequest()
    if "model" in obj:
        req.model = obj["model"]
    if "prompt" in obj:
        req.prompt = obj["prompt"]
    if "messages" in obj:
        req.messages = []
        for msg in obj["messages"]:
            req.messages.append(msg)
    if "frequency_penalty" in obj:
        req.frequency_penalty = float(obj["frequency_penalty"])
    if "repetition_penalty" in obj:
        req.repetition_penalty = float(obj["repetition_penalty"])
    if "presence_penalty" in obj:
        req.presence_penalty = float(obj["presence_penalty"])
    if "temperature" in obj:
        req.temperature = float(obj["temperature"])
    if "top_p" in obj:
        req.top_p = float(obj["top_p"])
    if "min_p" in obj:
        req.min_p = float(obj["min_p"])
    if "top_k" in obj:
        req.top_k = int(obj["top_k"])
    if "max_tokens" in obj:
        req.max_tokens = int(obj["max_tokens"])
    if "n" in obj:
        req.n = int(obj["n"])
    if "stop" in obj:
        req.stop = obj["stop"]
    if "ignore_eos" in obj:
        req.ignore_eos = bool(obj["ignore_eos"])
    return req

def load_from_csv(file_path:str):
    max_requests = 0 ## 0 means all
    raw_requests = []
    with open(args.requests_file, "r") as f:
        try:
            reader = csv.DictReader(f)
            line = next(reader)
            idx = 0
            while line and (max_requests<=0 or idx < max_requests):
                line2 = {decode_unicode_strings(k):v for k,v in line.items()}
                if "request_body" in line2:
                    body = line2["request_body"].strip()
                    body = re.sub(r'\\(.)', r'\1', body)
                    obj = json.loads(body)
                    req = create_request(obj)
                    raw_requests.append(req)
                idx+=1
                line = next(reader)
        except StopIteration:
            logger.info(f"EOS of file: {args.requests_file}")
    return raw_requests

def main(args):
    if not args.requests_file or not os.path.exists(args.requests_file) or not os.path.isfile(args.requests_file):
        raise ValueError("Invalid requests-file")
    logger.info(f"Loading request data from {args.requests_file}")
    raw_requests = []
    if args.requests_file.endswith(".csv"):
        raw_requests = load_from_csv(args.requests_file)
    if len(raw_requests) == 0:
        logger.info("No requests loaded")
        return
    ## analyze requests
    logger.info(f"Get num of requests: {len(raw_requests)}")
    #for i in range(len(raw_requests)):
    #    print(f"request[{i}]: {raw_requests[i]}")
    if args.tokenizer and os.path.exists(args.tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompt_lens = []
        for req in raw_requests:
            req.prompt_len = len(tokenizer.encode(req.prompt)) if req.prompt else 0
            prompt_lens.append(req.prompt_len)
        prompt_lens_percent = np.percentile(prompt_lens, [50, 90, 99])
        prompt_lens_avg = np.mean(prompt_lens)
        logger.info(f"Request prompt length(avg,p50,p90,p99): {prompt_lens_avg:.1f}, {prompt_lens_percent[0]:.1f}, {prompt_lens_percent[1]:.1f}, {prompt_lens_percent[2]:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw requests from file")
    parser.add_argument("--requests-file", type=str, help="The cvs file path which contains original client requests data")
    parser.add_argument("--tokenizer", type=str, help="Optional, if set, use it to calualate the input lengh")
    args = parser.parse_args()
    main(args)

