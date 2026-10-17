"""Retrieval quality on the test questions (EXP-07, EXP-08).

A hit means a chunk of the question's source document is retrieved. Reports
recall@k and MRR for each retriever configuration. Unanswerable examples are
skipped: their source document is in the corpus, so they are answerable by RAG.
"""

import argparse
import json
from pathlib import Path

from rag_retrieval import MODES, Retriever

DEFAULT_KS = (1, 3, 5, 10)


def first_hit_rank(results, document_id):
    """1-based rank of the first chunk from document_id, or None."""
    for rank, chunk in enumerate(results, start=1):
        if chunk['document_id'] == document_id:
            return rank
    return None


def retrieval_metrics(ranks, ks=DEFAULT_KS):
    n = len(ranks)
    metrics = {f'recall@{k}': round(sum(1 for r in ranks if r is not None and r <= k) / n, 4) if n else 0.0
               for k in ks}
    metrics['mrr'] = round(sum(1 / r for r in ranks if r is not None) / n, 4) if n else 0.0
    metrics['queries'] = n
    return metrics


def load_queries(path):
    with path.open('r', encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    seen, queries = set(), []
    for record in records:
        key = (record['document_id'], record['instruction'])
        if record.get('answerable', True) and key not in seen:
            seen.add(key)
            queries.append(record)
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', type=Path, required=True, help='test.jsonl')
    parser.add_argument('--index-dir', type=Path, default=Path('models/rag_index'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/retrieval.json'))
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    parser.add_argument('--rerank', action='store_true', help='Also evaluate each mode with the reranker.')
    parser.add_argument('--candidates', type=int, default=50)
    args = parser.parse_args()

    queries = load_queries(args.input)
    retriever = Retriever(args.index_dir)
    depth = max(DEFAULT_KS)
    configs = [(mode, False) for mode in args.modes]
    if args.rerank:
        configs += [(mode, True) for mode in args.modes]

    results = {}
    for mode, rerank in configs:
        name = f'{mode}+rerank' if rerank else mode
        ranks = [first_hit_rank(retriever.search(q['instruction'], depth, mode, rerank, args.candidates), q['document_id'])
                 for q in queries]
        results[name] = retrieval_metrics(ranks)
        print(name, json.dumps(results[name], ensure_ascii=False))

    summary = {'index': retriever.meta, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
