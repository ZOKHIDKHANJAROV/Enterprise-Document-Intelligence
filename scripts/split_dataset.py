import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_records(path):
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Split examples by document id.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/dataset'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.90)
    parser.add_argument('--validation-ratio', type=float, default=0.05)
    parser.add_argument('--max-documents', type=int, default=None)
    args = parser.parse_args()

    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError('train_ratio + validation_ratio must be < 1')

    records = load_records(args.input)
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get('document_id', 'unknown'))].append(record)

    ids = list(groups)
    random.Random(args.seed).shuffle(ids)
    if args.max_documents:
        ids = ids[:args.max_documents]

    n = len(ids)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.validation_ratio)
    train_ids = set(ids[:train_end])
    val_ids = set(ids[train_end:val_end])
    test_ids = set(ids[val_end:])

    train = [r for r in records if str(r.get('document_id')) in train_ids]
    validation = [r for r in records if str(r.get('document_id')) in val_ids]
    test = [r for r in records if str(r.get('document_id')) in test_ids]

    write_jsonl(args.output_dir / 'all.jsonl', records)
    write_jsonl(args.output_dir / 'train.jsonl', train)
    write_jsonl(args.output_dir / 'validation.jsonl', validation)
    write_jsonl(args.output_dir / 'test.jsonl', test)

    summary = {'documents': n, 'examples': len(records), 'train_examples': len(train), 'validation_examples': len(validation), 'test_examples': len(test), 'train_documents': len(train_ids), 'validation_documents': len(val_ids), 'test_documents': len(test_ids)}
    (args.output_dir / 'split_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
