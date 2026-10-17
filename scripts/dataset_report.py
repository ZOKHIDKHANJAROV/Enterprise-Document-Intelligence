"""EXP-00 dataset sanity report.

Checks the QA dataset before training: empty fields, source documents present
in the corpus, duplicate questions, document-level split integrity, lengths,
document-type and grounding distributions. Writes a random sample to a
Markdown file for manual review.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SPLITS = ('train', 'validation', 'test')


def load_jsonl(path):
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    text = re.sub(r'[^\w\s]', ' ', (text or '').lower().replace('ё', 'е'))
    return re.sub(r'\s+', ' ', text).strip()


def length_stats(values):
    if not values:
        return {'mean': 0, 'median': 0, 'max': 0}
    ordered = sorted(values)
    return {'mean': round(sum(ordered) / len(ordered), 1), 'median': ordered[len(ordered) // 2], 'max': ordered[-1]}


def build_report(records, splits=None, corpus_ids=None):
    """Returns (report, problems). problems lists issues that should block training."""
    problems = []
    empty = sum(1 for r in records if not str(r.get('instruction', '')).strip() or not str(r.get('output', '')).strip())
    missing_id = sum(1 for r in records if not str(r.get('document_id') or '').strip())
    if empty:
        problems.append(f'{empty} records with empty instruction or output')
    if missing_id:
        problems.append(f'{missing_id} records without document_id')

    questions = Counter(normalize(r.get('instruction')) for r in records)
    duplicate_questions = sum(count - 1 for count in questions.values() if count > 1)
    pairs = Counter((str(r.get('document_id')), normalize(r.get('instruction'))) for r in records)
    duplicate_in_document = sum(count - 1 for count in pairs.values() if count > 1)

    document_ids = {str(r.get('document_id')) for r in records}
    per_document = Counter(str(r.get('document_id')) for r in records)
    report = {
        'examples': len(records),
        'documents': len(document_ids),
        'examples_per_document': length_stats(list(per_document.values())),
        'empty_records': empty,
        'missing_document_id': missing_id,
        'duplicate_question_rate': round(duplicate_questions / len(records), 4) if records else 0.0,
        'duplicate_questions_within_document': duplicate_in_document,
        'instruction_chars': length_stats([len(r.get('instruction') or '') for r in records]),
        'input_chars': length_stats([len(r.get('input') or '') for r in records]),
        'output_chars': length_stats([len(r.get('output') or '') for r in records]),
        'document_types': dict(Counter(r.get('document_type', 'unknown') for r in records).most_common()),
    }

    grounding = [r['grounding'] for r in records if isinstance(r.get('grounding'), (int, float))]
    if grounding:
        report['grounding'] = {**length_stats(grounding), 'min': min(grounding),
                               'below_0.8': sum(1 for g in grounding if g < 0.8)}

    if corpus_ids is not None:
        missing_sources = sorted(document_ids - corpus_ids)
        report['missing_source_documents'] = len(missing_sources)
        if missing_sources:
            problems.append(f'{len(missing_sources)} document_ids not found in corpus, e.g. {missing_sources[:3]}')

    if splits:
        ids = {name: {str(r.get('document_id')) for r in rows} for name, rows in splits.items()}
        report['splits'] = {name: {'examples': len(rows), 'documents': len(ids[name])} for name, rows in splits.items()}
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                if first in ids and second in ids:
                    leaked = ids[first] & ids[second]
                    if leaked:
                        problems.append(f'{len(leaked)} documents in both {first} and {second}')
        for name, rows in splits.items():
            if not rows:
                problems.append(f'{name} split is empty')

    report['problems'] = problems
    return report, problems


def review_sample(records, size, seed):
    sample = random.Random(seed).sample(records, min(size, len(records)))
    lines = ['# QA review sample', '',
             'Для каждого примера проверьте: ответ следует из контекста, вопрос осмысленный, нет выдумок.', '']
    for index, record in enumerate(sample, start=1):
        lines += [f"## {index}. {record.get('document_type', 'unknown')} · `{record.get('document_id')}`", '',
                  f"**Вопрос:** {record.get('instruction')}", '',
                  f"**Ответ:** {record.get('output')}", '']
        if 'grounding' in record:
            lines += [f"**Grounding:** {record['grounding']}", '']
        lines += ['**Контекст:**', '', '> ' + (record.get('input') or '').replace('\n', '\n> '), '',
                  '- [ ] верно  - [ ] неточно  - [ ] выдумано', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', type=Path, required=True, help='QA JSONL (e.g. qa_raw.jsonl or all.jsonl).')
    parser.add_argument('--corpus', type=Path, help='corpus.jsonl, to check that source documents exist.')
    parser.add_argument('--split-dir', type=Path, help='Directory with train/validation/test.jsonl.')
    parser.add_argument('--output', type=Path, default=Path('evaluation/dataset_report.json'))
    parser.add_argument('--sample-output', type=Path, default=Path('evaluation/qa_review_sample.md'))
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strict', action='store_true', help='Exit with code 1 if problems are found.')
    args = parser.parse_args()

    records = load_jsonl(args.input)
    corpus_ids = {str(d['document_id']) for d in load_jsonl(args.corpus)} if args.corpus else None
    splits = None
    if args.split_dir:
        splits = {name: load_jsonl(args.split_dir / f'{name}.jsonl') for name in SPLITS
                  if (args.split_dir / f'{name}.jsonl').exists()}

    report, problems = build_report(records, splits, corpus_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_output.write_text(review_sample(records, args.sample_size, args.seed), encoding='utf-8')

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report: {args.output}\nReview sample: {args.sample_output}')
    if problems and args.strict:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
