import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from format_sft_dataset import SYSTEM_PROMPT, merge_system_prompt  # noqa: E402  same prompt as training
from model_config import DEFAULT_CONFIG, load_config, load_for_inference, resolve_adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='QLoRA config; its model_name is the base model.')
    parser.add_argument('--base-model', default=None, help='Override model_name from --config.')
    parser.add_argument('--adapter', default=None, help='LoRA adapter directory; omit to chat with the base model.')
    parser.add_argument('--with-adapter', action='store_true', help='Use the adapter at output_dir from --config.')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter = resolve_adapter(cfg, args.adapter, args.with_adapter)
    tokenizer, model = load_for_inference(cfg, args.base_model, adapter)

    print(f"{'QLoRA' if adapter else 'Base'} assistant ready. Type 'exit' to stop.")
    while True:
        question = input('\nUser: ').strip()
        if question.lower() in {'exit', 'quit'}:
            break
        if not question:
            continue
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
        ]
        if cfg.get('merge_system_prompt'):
            messages = merge_system_prompt(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        print('Assistant:', tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == '__main__':
    main()
