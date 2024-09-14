import os
import json
import random
import requests

from loguru import logger
from tqdm import tqdm
from typing import List, Optional, Tuple
from datasets import Dataset, load_dataset, VerificationMode
from transformers import AutoTokenizer

HF_DATASET_PRESET = {
    "sharegpt_wizard": "Thermostatic/ShareGPT_wizard_vicuna_unfiltered_no_alignment",
    "sharegpt_vicuna": "Aeala/ShareGPT_Vicuna_unfiltered",
    "cnn_dailymail": "cnn_dailymail:3.0.0",
    "dolly": "databricks/databricks-dolly-15k",
    "alpaca": "yahma/alpaca-cleaned",
    "alpaca_code": "iamtarun/python_code_instructions_18k_alpaca",
}

CNN_DAILYMAIL_SYS_PROMPT = "As an AI assist, could you please summarize or highlight the following content "
ALPACA_CODE_SYS_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

def get_hf_dataset_path(key: str) -> str:
    if key in HF_DATASET_PRESET:
        return HF_DATASET_PRESET[key]
    return None

def load_hf_dataset(ds_path) -> Dataset:
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            dataset = json.load(f)
    else:
        logger.info(f"Downloading dataset from huggingface: {ds_path}")
        parts = ds_path.split(":")
        if len(parts) == 1:
            dataset = load_dataset(ds_path, split="train", verification_mode=VerificationMode.NO_CHECKS)
        elif len(parts) == 2:
            data_name, subset_name = parts
            dataset = load_dataset(data_name, subset_name, split="test")
        else:
            raise RuntimeError("Dataset name is in invalid format. (valid fmt: '<dataset_name>' or '<dataset_name>:<subset_name>')")
    return dataset

def load_sharegpt_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load ShareGPT dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_cnn_dailymail_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load CNN_Dailymail dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(CNN_DAILYMAIL_SYS_PROMPT + data["article"], data["highlights"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_dolly_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Dolly dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(data["instruction"], data["response"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_alpaca_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Alpaca dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(data["instruction"] + " " + data["input"], data["output"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_alpaca_code_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Alpaca coding dataset and return list of tuple(instruct, input, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    result = []
    for data in dataset:
        prompt = ALPACA_CODE_SYS_PROMPT + " ### Instruction: " + data["instruction"] + " ### Input: " + data["input"]
        result.append((prompt, data["output"]))
    random.shuffle(result)
    return result

def load_requests_from_json(tokenizer:AutoTokenizer, path:str, num_reqs:int, in_min_len:List[int], in_max_len:List[int], out_min_len:List[int], out_max_len:List[int]) -> List[Tuple[str, int, int]]:
    output = []
    if tokenizer is None:
        raise RuntimeError("Invalid tokenizer")
    if not os.path.exists(path):
        raise RuntimeError(f"Not exist dataset: {path}")
    if num_reqs <= 0 or num_reqs != len(in_min_len) or num_reqs != len(in_max_len) or num_reqs != len(out_min_len) or num_reqs != len(out_max_len):
        raise RuntimeError(f"Invalid argument: {num_reqs}, {len(in_min_len)}, {len(in_max_len)}, {len(out_min_len)}, {len(out_max_len)}")
    logger.info(f"Load from dataset: {path}")
    dataset = []
    with open(path, "r") as f:
        data = json.load(f)
    if "kind" in data and data["kind"] == "ppio-internal":
        for d in data["data"]:
            dataset.append((d["prompt"], d["output"]))
    elif "sharegpt" in path.lower() or "share_gpt" in path.lower():
        dataset = [d for d in data if len(d["conversations"]) >= 2]
        dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset if len(data["conversations"][0]["value"]) > 10 and len(data["conversations"][1]["value"]) > 10]
    logger.info(f"The dataset has {len(dataset)} samples")
    if len(dataset) == 0:
        return output
    random.shuffle(dataset)
    pb = tqdm(total=num_reqs, smoothing=0.0)

    def _adjust_prompt(tokenizer, prompt, output, min_len, max_len, min_len_o, max_len_o):
        if min_len is None or min_len_o is None:
            return None, None, None
        max_len = min_len if max_len is None else max_len
        max_len_o = min_len_o if max_len_o is None else max_len_o
        prompt_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(output)
        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)
        if prompt_len < min_len:
            return None, None, None
        if prompt_len > max_len:
            prompt_len = random.randint(min_len, max_len)
            prompt = tokenizer.decode(prompt_tokens[0:prompt_len])
        if output_len > max_len_o or output_len < min_len_o:
            output_len = random.randint(min_len_o, max_len_o)
        return prompt, prompt_len, output_len

    for i in range(len(dataset)):
        if len(output) >= num_reqs:
            break
        data = dataset[i]
        n = len(output)
        prompt, prompt_len, output_len = _adjust_prompt(tokenizer, data[0], data[1], in_min_len[n], in_max_len[n], out_min_len[n], out_max_len[n])
        if prompt is not None:
            output.append((prompt, prompt_len, output_len))
            pb.update(1)
    pb.close()
    return output

def get_model(url: str, headers = None)->Optional[str]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    return model_list[0]["id"] if model_list else None

