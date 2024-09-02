import os
import time
import asyncio
import aiohttp
import argparse
import json
import random
import urllib.parse
from typing import List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
## pip install llama-index llama-index-embeddings-huggingface
from llama_index.core.evaluation import (CorrectnessEvaluator, SemanticSimilarityEvaluator)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.base.embeddings.base import (BaseEmbedding, SimilarityMode, similarity,)

from async_request_sender import RequestData, Response, InputParameter, AysncRequestSender
import util

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
CONCURRENCY=8
MIN_INPUT_LEN = 1000
#TEST_DS_NAMES = ["sharegpt_wizard", "sharegpt_vicuna", "cnn_dailymail", "dolly", "alpaca", "alpaca_code"]
TEST_DS_NAMES = ["sharegpt_vicuna"]
PRINT_SAMPLES = 1
PRINT_REPONSES = 10

@dataclass
class EmbeddingOutput:
    embeddings: List[float] = field(default_factory=list)
    prompt_len: int = 0
    total_len: int = 0
    err: str = ""

@dataclass
class SimilarityOutput:
    prompt: str,
    reference: str,
    generation: str,
    score: float

@dataclass
class DatasetElement:
    prompt: str
    prompt_tokens: List[int]
    response: str
    response_tokens: List[int]

def load_samples_from_dataset(tokenizer: AutoTokenizer, num_samples, dataset) -> List[DatasetElement]:
    if dataset is not None:
        dataset_names = dataset.split(",")
    if dataset_names is None:
        dataset_names = [util.get_hf_dataset_path(name) for name in TEST_DS_NAMES]
    num_ds = len(dataset_names)
    reqs_per_ds = num_samples // num_ds
    if reqs_per_ds * num_ds < num_samples:
        reqs_per_ds += 1
    logger.info(f"Loading {reqs_per_ds} * {num_ds} requests from datasets: {dataset_names}")
    samples: List[DatasetElement] = []
    sub_samples: List[List[DatasetElement]] = [[] for _ in range(len(dataset_names))]
    idx = 0
    for name in dataset_names:
        logger.info(f"Load dataset: {name}")
        dataset = None
        if "sharegpt" in name.lower() or "share_gpt" in name.lower():
            dataset = util.load_sharegpt_dataset(name)
        elif "cnn_dailymail" in name.lower():
            dataset = util.load_cnn_dailymail_dataset(name)
        elif "dolly" in name.lower():
            dataset = util.load_dolly_dataset(name)
        elif "alpaca" == name.lower():
            dataset = util.load_alpaca_dataset(name)
        elif "_code_" == name.lower():
            dataset = util.load_alpaca_code_dataset(name)
        if dataset is None:
            idx += 1
            continue
        for data in dataset:
            if len(sub_samples[idx]) > reqs_per_ds * 4:
                break
            prompt = data[0]
            response = data[1]
            prompt_tokens = tokenizer.encode(prompt)
            response_tokens = tokenizer.encode(response)
            if len(prompt_tokens) >= MIN_INPUT_LEN:
                sub_samples[idx].append(DatasetElement(prompt=prompt, prompt_tokens=prompt_tokens, response=response, response_tokens=response_tokens))
        logger.info(f"Got {len(sub_samples[idx])} samples from dataset")
        idx += 1
    sub_samples= sorted(sub_samples, key=lambda x: len(x))
    more = 0
    for i in range(len(sub_samples)):
        num = min(reqs_per_ds+more, len(sub_samples[i]))
        samples = samples + sub_samples[i][:num]
        more = reqs_per_ds * (i+1) - len(samples)
    if len(samples) > num_samples:
        samples = samples[:num_samples]
    random.shuffle(samples)
    return samples

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

def evaluate_similarity(url:str, model_name:str, api_key: str, responses: List[Response]):
    results:List[SimilarityOutput] = []
    for res in responses:
        prompt = res.prompt
        ref = res.ref_output
        gen = res.generated
        score = 0.0
        embeddings = asyncio.run(async_generate_embeddings(url, gen, model_name, api_key)) # url, prompt, model_name, api_key
        if embeddings is None:
            raise RuntimeError(f"Failed to generate embeddings vector for {url}, {model_name}, with text: {gen}")
        e1 = result.embeddings
        embeddings = asyncio.run(async_generate_embeddings(url, ref, model_name, api_key)) # url, prompt, model_name, api_key
        if embeddings is None:
            raise RuntimeError(f"Failed to generate embeddings vector for {url}, {model_name}, with text: {ref}")
        e2 = result.embeddings
        score = similarity(e1, e2, SimilarityMode.DEFAULT) ## mode: SimilarityMode.DEFAULT, SimilarityMode.EUCLIDEAN, SimilarityMode.DOT_PRODUCT
        results.append(SimilarityOutput(prompt=prompt, reference=ref, generation=gen, score=score))

def main(args: argparse.Namespace):
    if not args.endpoint:
        raise ValueError("Invalid LLM endpoint")
    if not args.model_name:
        raise ValueError("Invalid model name")
    is_chat = "chat/completions" in args.endpoint
    url = urllib.parse.urlparse(args.endpoint)
    if len(url.path) == 0:
        raise ValueError(f"Invalid endpoint: {args.endpoint}")

    ## initialize tokenizer and sender
    prefix = args.endpoint[0:args.endpoint.find(url.path)]
    sender = AysncRequestSender("", prefix, "", "/v1/models", url.path, url.path, None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.tokenizer is None or not os.path.exists(args.tokenizer):
        raise ValueError(f"The tokenizer is not valid: {args.tokenizer}")

    ## check health
    if not args.ignore_check:
        health_url = prefix + "/health"
        logger.info(f"Checking health via URL: {health_url}")
        if not sender.check_health(4):
            raise RuntimeError(f"Failed to check LLM server via {health_url}")
        str_models = sender.get_models()
        logger.info(f"Model names: {str_models}")
        if str_models is None or "\""+args.model_name+"\"" not in str_models:
            raise RuntimeError(f"Failed to check model names: {args.model_name}")

    ## load samples from dataset
    samples = load_samples_from_dataset(tokenizer, args.num_requests, args.dataset)
    logger.info(f"Loaded {len(samples)} samples from dataset")
    if PRINT_SAMPLES > 0:
        for i in range(PRINT_SAMPLES):
            logger.info(f"Sample[{i}]({len(samples[i].prompt_tokens)}, {len(samples[i].response_tokens)}): {samples[i].prompt} ===> {samples[i].response}")

    parameters = InputParameter(model=args.model_name, n=1, presence_penalty=0, frequency_penalty=0, repetition_penalty=1, temperature=0, top_p=1, stream=True)
    batch_requests = [RequestData(prompt=s.prompt, prompt_len=len(s.prompt_tokens), ref_output=s.response, max_tokens=len(s.response_tokens)) for s in samples]
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(CONCURRENCY, batch_requests, parameters, is_chat))
    end_time = time.perf_counter()
    logger.info(f"Done sending {len(batch_requests)} requests in {(end_time-start_time):.3f} seconds")
    evaluate_similarity(args.embedding_url, args.embedding_model, args.embedding_api_key, sender.get_response())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM engine stablity tests")
    # LLM Server
    parser.add_argument("--endpoint", type=str, help="The http URL to call LLM service")
    parser.add_argument("--tokenizer", type=str, help="The tokenizer model path used to encode & decode")
    parser.add_argument("--dataset", type=str, help="The dataset name or local path, can be multiple dataset, separated with ','")
    parser.add_argument("--model-name", type=str, help="The model name to call")
    parser.add_argument("--num-requests", type=int, default=1000, help="The total num of requests used for test")
    parser.add_argument("--embedding-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", help="The embedding servering URL")
    parser.add_argument("--embedding-model", type=str,  default="text-embedding-v1", help="The embedding model name")
    parser.add_argument("--embedding-api-key", type=str, help="The api key to invoke the embedding server")
    parser.add_argument("--ignore-check", action="store_true", help="If set, do not check the engine health")
    args = parser.parse_args()
    main(args)

