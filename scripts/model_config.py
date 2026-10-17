"""Shared access to configs/qlora*.yaml for training, prediction and chat."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path('configs/qlora.yaml')


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def base_model_name(cfg, override=None):
    """--base-model wins over the config, so a run can be pointed elsewhere explicitly."""
    return override or cfg['model_name']


def resolve_adapter(cfg, adapter=None, with_adapter=False):
    """Explicit --adapter, else the config's output_dir with --with-adapter, else None (base model)."""
    if adapter:
        return adapter
    return cfg['output_dir'] if with_adapter else None


def load_for_inference(cfg, base_model=None, adapter=None):
    """4-bit base model (plus optional LoRA adapter) with the config's quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = cfg['quantization']
    compute_dtype = getattr(torch, q['compute_dtype'])
    if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        compute_dtype = torch.float16
    name = base_model_name(cfg, base_model)
    quant = BitsAndBytesConfig(load_in_4bit=q['load_in_4bit'], bnb_4bit_quant_type=q['quant_type'],
                               bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=q['double_quant'])
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map='auto', torch_dtype=compute_dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model
