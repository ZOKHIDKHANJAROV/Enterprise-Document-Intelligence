"""Add unanswerable examples to each split, in place.

A question from one document is paired with the context of another document
from the same split, and the expected answer is a refusal. This teaches the
adapter to say that the context is insufficient instead of guessing, and lets
evaluation measure that. Pairs stay inside a split, so no document leaks
across train/validation/test. Re-running replaces previously added examples.
"""

import argparse
import json
import random
import re
from pathlib import Path

UNANSWERABLE_ANSWER = 'В предоставленном контексте недостаточно информации, чтобы ответить на этот вопрос.'
# A donor context is rejected if it shares this much of the question/answer
# vocabulary, since it might actually contain the answer.
MAX_OVERLAP = 0.2
SPLITS = ('train', 'validation', 'test')

REFUSAL_PATTERN = re.compile(
    r'недостаточно (?:информации|сведений|данных)'
    r'|(?:информации|сведений|данных) недостаточно'
    r'|нет (?:информации|сведений|данных)'
    r'|не содерж\w* (?:информации|сведений|данных|ответа)',
    re.IGNORECASE)


def is_refusal(text):
    return bool(REFUSAL_PATTERN.search((text or '').replace('ё', 'е')))


def stems(text):
    words = re.sub(r'[^\w\s]', ' ', (text or '').lower().replace('ё', 'е')).split()
    return {word[:5] for word in words if len(word) >= 4}


def overlap(record, context):
    target = stems(record['instruction']) | stems(record['output'])
    return len(target & stems(context)) / len(target) if target else 1.0


def make_unanswerable(records, ratio, seed, attempts=20):
    """Return new refusal examples for about ratio * len(answerable) questions."""
    answerable = [r for r in records if r.get('answerable', True)]
    contexts = {}
    for record in answerable:
        if record.get('input'):
            contexts.setdefault(record['document_id'], []).append(record)
    document_ids = sorted(contexts)
    if len(document_ids) < 2:
        return []

    rng = random.Random(seed)
    sources = rng.sample(answerable, min(len(answerable), round(len(answerable) * ratio)))
    added = []
    for source in sources:
        for _ in range(attempts):
            donor_id = rng.choice(document_ids)
            if donor_id == source['document_id']:
                continue
            donor = rng.choice(contexts[donor_id])
            if overlap(source, donor['input']) <= MAX_OVERLAP:
                added.append({
                    'document_id': donor['document_id'],
                    'document_type': donor.get('document_type', 'unknown'),
                    'instruction': source['instruction'],
                    'input': donor['input'],
                    'output': UNANSWERABLE_ANSWER,
                    'answerable': False,
                    'question_document_id': source['document_id'],
                })
                break
    return added


def load_jsonl(path):
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset-dir', type=Path, default=Path('data/dataset'))
    parser.add_argument('--ratio', type=float, default=0.1, help='Unanswerable examples per answerable example.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    summary = {}
    for offset, name in enumerate(SPLITS):
        path = args.dataset_dir / f'{name}.jsonl'
        if not path.exists():
            continue
        records = [r for r in load_jsonl(path) if r.get('answerable', True)]
        added = make_unanswerable(records, args.ratio, args.seed + offset)
        combined = records + added
        random.Random(args.seed + offset).shuffle(combined)
        write_jsonl(path, combined)
        summary[name] = {'answerable': len(records), 'unanswerable': len(added)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
