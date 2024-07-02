from loguru import logger
import json
from tqdm import tqdm
import numpy as np
import random
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


POLICY_KEYS = {
    "nature": {
        "max_turns",
        "min_prompt_len",
        "min_output_len",
        "max_prompt_len",
        "max_prompt_output_len",
    },
    "fixed": {"fixed_prompt_len", "fixed_output_len"},
    "normal": {
        "max_seq_len",
        "prompt_len_mean",
        "prompt_len_std",
        "output_len_mean",
        "output_len_std",
    },
}

class DatasetSampler:
    def __init__(self, dataset_path: str):
        logger.info(f"loading dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"{len(dataset)} samples loaded")

        self.samples = []
        for sample in tqdm(dataset):
            conversions = []
            for conversion in sample["conversations"]:
                if len(conversion["value"]) == 0:
                    continue
                elif len(conversions) % 2 == 0:
                    if conversion["from"] in ["user", "human"]:
                        conversions.append(conversion["value"])
                else:
                    if conversion["from"] in ["gpt", "chatgpt", "bing", "bard"]:
                        conversions.append(conversion["value"])
            if len(conversions) % 2 == 1:
                conversions = conversions[:-1]
            if len(conversions) > 0:
                self.samples.append(conversions)
        logger.info(f"{len(self.samples)} are filtered")

    def sample_requests(self, num_warmup, num_test, tokenizer, system_prompt, policy, **kwargs):
        if len(self.samples) == 0:
            logger.error("The dataset is empty")
            return None, None
        keys = POLICY_KEYS[policy]
        if not keys.issubset(kwargs.keys()):
            logger.error(f"The sampling policy is not support: {policy}")
            return None, None

        num_requests = num_warmup + num_test
        pb = tqdm(total=num_requests, smoothing=0.0)
        output = []
        output_warmup = []
        output_test = []
        sys_msg = ""
        if system_prompt:
            sys_msg = f"<<SYS>>\n{system_prompt}<</SYS>>\n\n"
        if policy == "nature":
            max_turns = kwargs["max_turns"]
            min_prompt_len = kwargs["min_prompt_len"]
            min_output_len = kwargs["min_output_len"]
            max_prompt_len = kwargs["max_prompt_len"]
            max_prompt_output_len = kwargs["max_prompt_output_len"]
            logger.info(f"sampling with nature policy, max_turns: {max_turns}, min_prompt_len: {min_prompt_len}, min_output_len: {min_output_len}, max_prompt_len: {max_prompt_len}, max_prompt_output_len: {max_prompt_output_len}")
            permutation = np.random.permutation(len(self.samples))
            shuffled_data = [self.samples[i] for i in permutation]
            for data in shuffled_data:
                turns = min(len(data) // 2, max_turns)
                selected_turn = np.random.randint(0, turns)
                prompt = ""
                prompt_output = ""
                has_ascii = False
                for idx, msg in enumerate(data[: (selected_turn + 1) * 2]):
                    if any(ord(c) >= 128 for c in msg):
                        has_ascii = True
                        break
                    if idx % 2 == 0:
                        if idx == 0:
                            prompt += f"<s>[INST] {sys_msg}{msg} [/INST]"
                        else:
                            prompt += f"<s>[INST] {msg} [/INST]"
                    else:
                        if idx == selected_turn * 2 + 1:
                            prompt_output = prompt + f"{msg}</s>"
                        else:
                            prompt_output += f"{msg}</s>"
                if has_ascii:
                    continue
                prompt_tokens = tokenizer.encode(prompt)
                if len(prompt_tokens) > max_prompt_len or len(prompt_tokens) < min_prompt_len:
                    continue
                prompt_output_tokens = tokenizer.encode(prompt_output)
                if (len(prompt_output_tokens) > max_prompt_output_len or len(prompt_output_tokens) - len(prompt_tokens) < min_output_len):
                    continue
                output.append((prompt, len(prompt_tokens), len(prompt_output_tokens) - len(prompt_tokens)))
                pb.update(1)
                if len(output) == num_requests:
                    break
        elif policy == "fixed":
            fixed_prompt_len = kwargs["fixed_prompt_len"]
            fixed_output_len = kwargs["fixed_output_len"]
            logger.info(f"sampling with fixed policy, fixed_prompt_len: {fixed_prompt_len}, fixed_output_len: {fixed_output_len}")
            permutation = np.random.permutation(len(self.samples))
            shuffled_data = [self.samples[i] for i in permutation]
            for data in shuffled_data:
                prompt = ""
                for idx, msg in enumerate(data):
                    if idx % 2 == 0:
                        if idx == 0:
                            prompt += f"<s>[INST] {sys_msg}{msg} [/INST]"
                        else:
                            prompt += f"<s>[INST] {msg} [/INST]"
                    else:
                        prompt += f"{msg} </s>"
                prompt_tokens = tokenizer.encode(prompt)
                if len(prompt_tokens) < fixed_prompt_len:
                    continue
                prompt_tokens = prompt_tokens[:fixed_prompt_len]
                prompt = tokenizer.decode(prompt_tokens)
                output.append((prompt, fixed_prompt_len, fixed_output_len))
                pb.update(1)
                if len(output) == num_requests:
                    break
        elif policy == "normal":
            max_seq_len = kwargs["max_seq_len"]
            prompt_len_mean = kwargs["prompt_len_mean"]
            prompt_len_std = kwargs["prompt_len_std"]
            output_len_mean = kwargs["output_len_mean"]
            output_len_std = kwargs["output_len_std"]
            logger.info(f"sampling with normal policy, prompt_len: {prompt_len_mean}, {prompt_len_std}, output_len: {output_len_mean}, {output_len_std}, max_seq_len: {max_seq_len}")
            weights = [sum([len(msg) for msg in sample]) * len(self.samples) + idx for idx, sample in enumerate(self.samples)]
            sorted_indices = np.argsort(weights)
            sorted_data = [self.samples[i] for i in sorted_indices]
            prompt_lens = np.rint(np.random.normal(prompt_len_mean, prompt_len_std, size=num_requests)).astype(np.int64)
            output_lens = np.rint(np.random.normal(output_len_mean, output_len_std, size=num_requests)).astype(np.int64)
            sorted_indices = np.argsort(prompt_lens + output_lens)
            prompt_lens = prompt_lens[sorted_indices]
            output_lens = output_lens[sorted_indices]
            for data in sorted_data:
                prompt_len = prompt_lens[len(output)]
                output_len = output_lens[len(output)]
                if prompt_len <= 0:
                    prompt_len = 1
                if prompt_len > max_seq_len:
                    prompt_len = max_seq_len - 1
                if output_len <= 0:
                    output_len = 1
                if output_len + prompt_len > max_seq_len:
                    output_len = max_seq_len - prompt_len
                prompt = ""
                for idx, msg in enumerate(data):
                    if idx % 2 == 0:
                        if idx == 0:
                            prompt += f"<s>[INST] {sys_msg}{msg} [/INST]"
                        else:
                            prompt += f"<s>[INST] {msg} [/INST]"
                    else:
                        prompt += f"{msg} </s>"
                prompt_tokens = tokenizer.encode(prompt)
                #logger.info(f"====2: {prompt_len}, {len(prompt_tokens)}")
                if len(prompt_tokens) < prompt_len:
                    continue
                prompt_tokens = prompt_tokens[:prompt_len]
                prompt = tokenizer.decode(prompt_tokens)
                #logger.info("====3")
                output.append((prompt, int(prompt_len), int(output_len)))
                pb.update(1)
                if len(output) == num_requests:
                    permutation = np.random.permutation(len(output))
                    output = [output[i] for i in permutation]
                    break
        else:
            logger.error(f"Invalid sampling policy: {policy}")
        pb.close()
        # split the output to warmup and test set
        output_warmup = output[:num_warmup]
        output_test = output[num_warmup:]
        logger.info(f"Got {len(output_warmup)} requests for warmup and {len(output_test)} for testing")
        total_tokens = lambda requests: sum(prompt_len + output_len for _, prompt_len, output_len in requests)
        avg_prompt_len = lambda requests: np.mean([p for _, p, _ in requests])
        avg_generate_len = lambda requests: np.mean([g for _, _, g in requests])
        logger.info(f"Total tokens for warmup: {total_tokens(output_warmup)}, and for test: {total_tokens(output_test)}")
        logger.info(f"Avg tokens for warmup: {avg_prompt_len(output_warmup)}, and for test: {avg_prompt_len(output_test)}")
        logger.info(f"Avg generated tokens for warmup: {avg_generate_len(output_warmup)}, and for test: {avg_generate_len(output_test)}")
        return output_warmup, output_test



