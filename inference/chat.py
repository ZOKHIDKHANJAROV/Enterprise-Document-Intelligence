import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    print("QLoRA assistant ready. Type 'exit' to stop.")
    while True:
        question = input('\nUser: ').strip()
        if question.lower() in {'exit', 'quit'}:
            break
        if not question:
            continue
        messages = [
            {'role': 'system', 'content': 'Ты корпоративный AI-ассистент. Не выдумывай факты. Если данных недостаточно, скажи об этом.'},
            {'role': 'user', 'content': question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        print('Assistant:', tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == '__main__':
    main()
