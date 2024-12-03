from peft import PeftModel
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import math
from loguru import logger


def merge_lora_to_base_model(adapter_name_or_path):
    model_name_or_path = 'TinyLlama/TinyLlama-1.1B-Chat-v0.4'
    adapter_name_or_path = adapter_name_or_path
    save_path = '/root/Hypo2Trans/merge_lora/'
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_max_length = 256
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info(f'Change model_max_length from {orig_ctx_len} to {model_max_length}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    trainable_params_file = os.path.join(adapter_name_or_path, "trainable_params.bin")
    if os.path.isfile(trainable_params_file):
        model.load_state_dict(torch.load(trainable_params_file, map_location=model.device), strict=False)
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--adapter_name_or_path", type=str, required=True, help='/root/Hypo2Trans/wsj/checkpoint-2000')
    args = parser.parse_args()
    merge_lora_to_base_model(args.adapter_name_or_path)