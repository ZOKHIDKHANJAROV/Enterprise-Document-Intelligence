import argparse
from pathlib import Path

# Heavy ML libraries are imported inside main() so the data helpers below can
# be tested without torch/transformers installed.


def dtype_from_name(name):
    import torch

    values = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    if name not in values:
        raise ValueError(f'Unsupported dtype: {name}')
    return values[name]


def example_length(example, tokenizer):
    text = tokenizer.apply_chat_template(example['prompt'] + example['completion'], tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def fits_max_length(example, count_tokens, max_length):
    """True if prompt + full answer fit, so truncation never cuts the answer."""
    return count_tokens(example) <= max_length


def filter_by_length(dataset, tokenizer, max_length, name):
    before = len(dataset)
    dataset = dataset.filter(lambda ex: fits_max_length(ex, lambda e: example_length(e, tokenizer), max_length))
    dropped = before - len(dataset)
    print(f'{name}: kept {len(dataset)} of {before} examples, dropped {dropped} longer than {max_length} tokens')
    if len(dataset) == 0:
        raise RuntimeError(f'{name}: no examples fit max_seq_length={max_length}; lower --max-context-chars in format_sft_dataset.py')
    return dataset


def main():
    import torch
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/qlora.yaml'))
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()

    with args.config.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for QLoRA training.')
    if cfg['training'].get('bf16') and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bf16; set compute_dtype: float16, fp16: true, bf16: false in the config.')

    q = cfg['quantization']
    compute_dtype = dtype_from_name(q['compute_dtype'])
    bnb = BitsAndBytesConfig(load_in_4bit=q['load_in_4bit'], bnb_4bit_quant_type=q['quant_type'], bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=q['double_quant'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t = cfg['training']
    train_ds = load_dataset('json', data_files=str(args.train), split='train').select_columns(['prompt', 'completion'])
    val_ds = load_dataset('json', data_files=str(args.validation), split='train').select_columns(['prompt', 'completion'])
    train_ds = filter_by_length(train_ds, tokenizer, t['max_seq_length'], 'train')
    val_ds = filter_by_length(val_ds, tokenizer, t['max_seq_length'], 'validation')

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], quantization_config=bnb, torch_dtype=compute_dtype, device_map='auto')
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    l = cfg['lora']
    peft_config = LoraConfig(r=l['r'], lora_alpha=l['alpha'], lora_dropout=l['dropout'], bias=l['bias'], task_type='CAUSAL_LM', target_modules=l['target_modules'])

    training_args = SFTConfig(output_dir=cfg['output_dir'], num_train_epochs=t['num_train_epochs'], per_device_train_batch_size=t['per_device_train_batch_size'], per_device_eval_batch_size=t['per_device_eval_batch_size'], gradient_accumulation_steps=t['gradient_accumulation_steps'], gradient_checkpointing=t['gradient_checkpointing'], learning_rate=t['learning_rate'], weight_decay=t['weight_decay'], fp16=t.get('fp16', False), bf16=t.get('bf16', False), logging_steps=t['logging_steps'], eval_strategy='steps', eval_steps=t['eval_steps'], save_strategy='steps', save_steps=t['save_steps'], save_total_limit=t['save_total_limit'], optim=t['optim'], warmup_ratio=t['warmup_ratio'], lr_scheduler_type=t['lr_scheduler_type'], report_to='none', load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, max_length=t['max_seq_length'], completion_only_loss=True)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(cfg['output_dir'])
    tokenizer.save_pretrained(cfg['output_dir'])
    print(f"Adapter saved to: {cfg['output_dir']}")


if __name__ == '__main__':
    main()
