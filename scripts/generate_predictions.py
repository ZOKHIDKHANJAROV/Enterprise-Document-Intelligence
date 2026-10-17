import argparse
import json
from pathlib import Path

from format_sft_dataset import DEFAULT_MAX_CONTEXT_CHARS, build_prompt_messages, truncate_context

# Heavy ML libraries are imported inside main() so the CLI and prompt building
# can be tested without torch/transformers installed.


def build_parser():
    parser = argparse.ArgumentParser(description='Generate test-set predictions. Omit --adapter for the base-model baseline (EXP-01).')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-model', default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-context-chars', type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    return parser


def main():
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = build_parser().parse_args()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, device_map='auto', torch_dtype=torch.float16)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model_label = args.adapter or f'base:{args.base_model}'
    print(f'Generating with: {model_label}')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open('r', encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for line in tqdm(source, desc='Generating predictions'):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = build_prompt_messages(record, args.max_context_chars)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            target.write(json.dumps({'document_id': record.get('document_id'), 'model': model_label, 'answerable': record.get('answerable', True), 'instruction': record['instruction'], 'context': truncate_context(record.get('input'), args.max_context_chars), 'reference': record['output'], 'prediction': prediction}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
