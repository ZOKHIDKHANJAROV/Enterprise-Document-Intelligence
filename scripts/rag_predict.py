"""RAG predictions with citations (EXP-07, EXP-08).

For each test question, retrieve chunks from the index, give them to the model
as numbered sources and ask it to cite them as [n]. Output rows match
generate_predictions.py, plus retrieved and cited document ids, so
evaluate_generation.py and judge_predictions.py work unchanged.
"""

import argparse
import json
import re
from pathlib import Path

from evaluate_retrieval import load_queries
from format_sft_dataset import build_prompt_messages, merge_system_prompt
from model_config import DEFAULT_CONFIG, base_model_name, load_config, load_for_inference, resolve_adapter
from rag_retrieval import MODES, Retriever

CITATION_NOTE = 'Ссылайся на источники номерами в квадратных скобках, например [1].'


def format_sources(chunks):
    return '\n\n'.join(f"[{n}] {chunk.get('file_name') or chunk['document_id']}\n{chunk['text']}"
                       for n, chunk in enumerate(chunks, start=1))


def build_rag_messages(question, chunks, merge_system=False):
    record = {'instruction': f'{question}\n\n{CITATION_NOTE}', 'input': format_sources(chunks)}
    messages = build_prompt_messages(record, max_context_chars=None)
    return merge_system_prompt(messages) if merge_system else messages


def parse_citations(prediction, chunks):
    """Document ids cited as [n] (also [1, 2] and [1][2]), in order, without duplicates."""
    cited = []
    for group in re.findall(r'\[(\d+(?:\s*[,;]\s*\d+)*)\]', prediction or ''):
        for number in re.split(r'\s*[,;]\s*', group):
            n = int(number)
            if 1 <= n <= len(chunks) and chunks[n - 1]['document_id'] not in cited:
                cited.append(chunks[n - 1]['document_id'])
    return cited


def unique(values):
    return list(dict.fromkeys(values))


def main():
    import torch
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', type=Path, required=True, help='test.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index-dir', type=Path, default=Path('models/rag_index'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--base-model', default=None)
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--with-adapter', action='store_true', help='Use the adapter at output_dir from --config.')
    parser.add_argument('--mode', default='hybrid', choices=MODES)
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = base_model_name(cfg, args.base_model)
    adapter = resolve_adapter(cfg, args.adapter, args.with_adapter)
    retriever = Retriever(args.index_dir)
    tokenizer, model = load_for_inference(cfg, base_model, adapter)
    retrieval = f"{args.mode}{'+rerank' if args.rerank else ''}@{args.top_k}"
    model_label = f'{base_model}+{adapter}' if adapter else f'base:{base_model}'
    print(f'Generating with: {model_label}, retrieval: {retrieval}')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open('w', encoding='utf-8') as target:
        for record in tqdm(load_queries(args.input), desc='RAG predictions'):
            chunks = retriever.search(record['instruction'], args.top_k, args.mode, args.rerank, args.candidates)
            messages = build_rag_messages(record['instruction'], chunks, cfg.get('merge_system_prompt'))
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            target.write(json.dumps({
                'document_id': record['document_id'], 'model': model_label, 'retrieval': retrieval,
                'answerable': True, 'instruction': record['instruction'], 'context': format_sources(chunks),
                'reference': record['output'], 'prediction': prediction,
                'retrieved_document_ids': unique(c['document_id'] for c in chunks),
                'cited_document_ids': parse_citations(prediction, chunks),
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
