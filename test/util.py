import os
import json
import random

from loguru import logger
from typing import List, Optional, Tuple
from datasets import Dataset, load_dataset, VerificationMode

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

