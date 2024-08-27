from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

#pretrained_model_dir = "/ppio/models/Meta-Llama-3-8B-Instruct"
#quantized_model_dir = "./model/Meta-Llama-3-8B-Instruct-FP8_"
pretrained_model_dir = "/ppio/models/Meta-Llama-3.1-8B-Instruct"
quantized_model_dir = "./model/Meta-Llama-3.1-8B-Instruct-FP8_"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize 512 dataset samples for calibration of activation scales
ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

# Define quantization config with static activation scales
quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",
    ignore_patterns=["re:.*lm_head"],
    kv_cache_quant_targets=("k_proj", "v_proj"),
)
# Load the model, quantize, and save checkpoint
model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)

