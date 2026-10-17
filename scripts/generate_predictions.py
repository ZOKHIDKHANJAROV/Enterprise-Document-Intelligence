import argparse
import json
from pathlib import Path

from format_sft_dataset import DEFAULT_MAX_CONTEXT_CHARS, build_prompt_messages, merge_system_prompt, truncate_context
from model_config import DEFAULT_CONFIG, base_model_name, load_config, load_for_inference, resolve_adapter

# Heavy ML libraries are imported inside main() so the CLI and prompt building
# can be tested without torch/transformers installed.


def build_parser():
    parser = argparse.ArgumentParser(description='Generate test-set predictions. Omit --adapter for the base-model baseline (EXP-01).')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='QLoRA config; its model_name is the base model.')
    parser.add_argument('--base-model', default=None, help='Override model_name from --config.')
    parser.add_argument('--adapter', default=None, help='LoRA adapter directory.')
    parser.add_argument('--with-adapter', action='store_true', help='Use the adapter at output_dir from --config.')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-context-chars', type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    return parser


def main():
    import torch
    from tqdm import tqdm

    args = build_parser().parse_args()
    cfg = load_config(args.config)
    base_model = base_model_name(cfg, args.base_model)
    adapter = resolve_adapter(cfg, args.adapter, args.with_adapter)
    tokenizer, model = load_for_inference(cfg, base_model, adapter)
    model_label = f'{base_model}+{adapter}' if adapter else f'base:{base_model}'
    print(f'Generating with: {model_label}')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open('r', encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for line in tqdm(source, desc='Generating predictions'):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = build_prompt_messages(record, args.max_context_chars)
            if cfg.get('merge_system_prompt'):
                messages = merge_system_prompt(messages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            target.write(json.dumps({'document_id': record.get('document_id'), 'model': model_label, 'answerable': record.get('answerable', True), 'instruction': record['instruction'], 'context': truncate_context(record.get('input'), args.max_context_chars), 'reference': record['output'], 'prediction': prediction}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
