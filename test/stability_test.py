import os
import time
import asyncio
import aiohttp
import argparse
import json
import random
import urllib.parse
import numpy as np
from typing import List, Optional, Tuple, Union
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
TEST_DS_NAMES = ["sharegpt_wizard", "sharegpt_vicuna", "cnn_dailymail", "dolly", "alpaca", "alpaca_code"]
PRINT_SAMPLES = 1
PRINT_REPONSES = 2
PRIMT_SIMILARITY = 2

@dataclass
class EmbeddingOutput:
    embeddings: List[float] = field(default_factory=list)
    prompt_len: int = 0
    total_len: int = 0
    err: str = ""

@dataclass
class SimilarityOutput:
    prompt: str = ""
    reference: str = ""
    generation: str = ""
    score: float = 0

@dataclass
class DatasetElement:
    source: str
    prompt: str
    prompt_tokens: Union[int, List[int]]
    response: str
    response_tokens: Union[int, List[int]]

def load_samples_from_dataset(tokenizer: AutoTokenizer, num_samples, dataset, min_prompt_len: int=10, min_output_len: int=10) -> List[DatasetElement]:
    dataset_names = None
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
            if len(prompt_tokens) >= min_prompt_len and len(response_tokens) >= min_output_len:
                sub_samples[idx].append(DatasetElement(source=name.lower(), prompt=prompt, prompt_tokens=prompt_tokens, response=response, response_tokens=response_tokens))
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
    #random.shuffle(samples)
    return samples

async def async_generate_embeddings(url: str, inputs: List[str], model_name: str, api_key: str = None) -> List[List[float]]:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "input": inputs,
            "model": model_name,
            "encoding_format": "float"
        }
        headers = {"User-Agent": "LLM API Client"}
        if api_key is not None and api_key != "":
            headers["Authorization"] = "Bearer " + api_key
        output: List[List[float]] = []
        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        data = json.loads(chunk_bytes)
                        if "data" in data and len(data["data"]) > 0:
                            for d in data["data"]:
                                output.append(d["embedding"])
                                #output = EmbeddingOutput(embeddings=data["data"][0]["embedding"], prompt_len=data["usage"]["prompt_tokens"], total_len=data["usage"]["total_tokens"])

                else:
                    logger.error(f"Failed to generate embedddings, error: {response.status}, {response.reason}")
        except Exception as ex:
            logger.error(f"Exception raised when generate embeddings: {ex}")
        return output

def evaluate_similarity(url:str, model_name:str, api_key: str, responses: List[Response]) -> List[SimilarityOutput]:
    results:List[SimilarityOutput] = []
    for res in responses:
        prompt = res.prompt
        ref = res.ref_output
        gen = res.generated
        score = 0.0
        embeddings = asyncio.run(async_generate_embeddings(url, [res.generated, res.ref_output], model_name, api_key)) # url, prompt, model_name, api_key
        if len(embeddings) == 2:
            e1 = embeddings[0]
            e2 = embeddings[1]
        else:
            continue
        """
        embeddings = asyncio.run(async_generate_embeddings(url, gen, model_name, api_key)) # url, prompt, model_name, api_key
        if embeddings is None:
            raise RuntimeError(f"Failed to generate embeddings vector for {url}, {model_name}, with text: {gen}")
        e1 = result.embeddings
        embeddings = asyncio.run(async_generate_embeddings(url, ref, model_name, api_key)) # url, prompt, model_name, api_key
        if embeddings is None:
            raise RuntimeError(f"Failed to generate embeddings vector for {url}, {model_name}, with text: {ref}")
        e2 = result.embeddings
        """
        score = similarity(e1, e2, SimilarityMode.DEFAULT) ## mode: SimilarityMode.DEFAULT, SimilarityMode.EUCLIDEAN, SimilarityMode.DOT_PRODUCT
        results.append(SimilarityOutput(prompt=prompt, reference=ref, generation=gen, score=score))
    return results

def save_dataset(samples: List[DatasetElement], file_path: str):
    data = {"kind": "ppio-internal", "data": []}
    for s in samples:
        data["data"].append({"source":s.source, "prompt_len": s.prompt_tokens if isinstance(s.prompt_tokens, int) else len(s.prompt_tokens), "output_len": s.response_tokens if isinstance(s.response_tokens, int) else len(s.response_tokens), "prompt": s.prompt, "output": s.response})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Dataset saved into file {file_path}")

def main(args: argparse.Namespace):
    if args.create_dataset:
        if args.tokenizer is None:
            raise ValueError("Invalid tokenizer")
        samples = load_samples_from_dataset(AutoTokenizer.from_pretrained(args.tokenizer), 2000, None, 100, 100)
        samples = sorted(samples, key=lambda e: len(e.prompt_tokens))
        logger.info(f"Loaded {len(samples)} samples from dataset")
        return save_dataset(samples, args.create_dataset)
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
    samples = []
    samples_internal = None
    if os.path.exists(args.dataset):
        with open(args.dataset, "r") as f:
            json_data= json.load(f)
        if "kind" in json_data and json_data["kind"] == "ppio-internal":
            samples_internal = []
            for data in json_data["data"]:
                samples_internal.append(DatasetElement(source=data["source"], prompt=data["prompt"], prompt_tokens=tokenizer.encode(data["prompt"]), response=data["output"], response_tokens=tokenizer.encode(data["output"])))
            samples = samples_internal[0:min(args.num_requests, len(samples_internal))]
            random.shuffle(samples)
            logger.info(f"Loaded samples from internal file {args.dataset}, got {len(samples)} samples")
    if len(samples) == 0:
        samples = load_samples_from_dataset(tokenizer, args.num_requests, args.dataset)
        logger.info(f"Loaded {len(samples)} samples from dataset: {args.dataset}")
    if samples is None or len(samples) == 0:
        raise RuntimeError(f"Failed to load samples from {args.dataset}")
    if PRINT_SAMPLES > 0:
        for i in range(PRINT_SAMPLES):
            logger.info(f"Sample[{i}]({len(samples[i].prompt_tokens)}, {len(samples[i].response_tokens)}): {samples[i].prompt} ===> {samples[i].response}")

    ## inference
    parameters = InputParameter(model=args.model_name, n=1, presence_penalty=0, frequency_penalty=0, repetition_penalty=1, temperature=0, top_p=1, stream=True)
    batch_requests = [RequestData(prompt=s.prompt, prompt_len=len(s.prompt_tokens), ref_output=s.response, max_tokens=len(s.response_tokens)) for s in samples]
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(CONCURRENCY, batch_requests, parameters, is_chat))
    end_time = time.perf_counter()
    logger.info(f"Done sending {len(batch_requests)} requests in {(end_time-start_time):.3f} seconds")
    responses = sender.get_response()
    if PRINT_REPONSES > 0:
        for i in range(min(PRINT_REPONSES, len(responses))):
            logger.info(f"[{i}] prompt: {responses[i].prompt}")
            logger.info(f"[{i}] reference: {responses[i].ref_output}")
            logger.info(f"[{i}] generated: {responses[i].generated}\n")

    ## evaluate
    similarity_ret = evaluate_similarity(args.embedding_url, args.embedding_model, args.embedding_api_key, responses)
    scores = []
    for i in range(len(similarity_ret)):
        logger.info(f"similarity[{i}]: {similarity_ret[i].score}")
        scores.append(similarity_ret[i].score)
        if i < PRIMT_SIMILARITY:
            logger.info(f"comparison[{i}]: \n###prompt: {similarity_ret[i].prompt}, \n### reference: {similarity_ret[i].reference}, \n### generation: {similarity_ret[i].generation}")
    p_scores = np.percentile(scores, [50, 70, 90])
    logger.info(f"samilarity: avg: {np.mean(scores):.3f}, min,max:{np.min(scores):.3f},{np.max(scores):.3f}, p50,p70,p90:{p_scores[0]:.3f},{p_scores[1]:.3f},{p_scores[2]:.3f}")
    ## update dataet
    if args.update_dataset and samples_internal is not None:
        if len(samples_internal) > len(responses):
            for i in range(len(responses)):
                r = responses[i]
                samples_internal[i] = DatasetElement(source=args.model_name, prompt=r.prompt, prompt_tokens=r.prompt_len, response=r.generated, response_tokens=r.output_len)
        else:
            samples_internal = []
            for i in range(len(responses)):
                r = responses[i]
                samples_internal.append(DatasetElement(source="", prompt=r.prompt, prompt_tokens=r.prompt_len, response=r.generated, response_tokens=r.output_len))
        save_dataset(samples_internal,args.update_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM engine stablity tests")
    # LLM Server
    parser.add_argument("--endpoint", type=str, help="The http URL to call LLM service")
    parser.add_argument("--tokenizer", type=str, help="The tokenizer model path used to encode & decode")
    parser.add_argument("--dataset", type=str, help="The dataset name or local path, can be multiple dataset, separated with ','")
    parser.add_argument("--model-name", type=str, help="The model name to call")
    parser.add_argument("--num-requests", type=int, default=100, help="The total num of requests used for test")
    parser.add_argument("--embedding-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", help="The embedding servering URL")
    parser.add_argument("--embedding-model", type=str,  default="text-embedding-v1", help="The embedding model name")
    parser.add_argument("--embedding-api-key", type=str, help="The api key to invoke the embedding server")
    parser.add_argument("--ignore-check", action="store_true", help="If set, do not check the engine health")
    ## update the local dataset
    parser.add_argument("--create-dataset", type=str, help="The path to local file to save the testing data")
    parser.add_argument("--update-dataset", type=str, help="The path to local file to update the testing data")
    args = parser.parse_args()
    main(args)

