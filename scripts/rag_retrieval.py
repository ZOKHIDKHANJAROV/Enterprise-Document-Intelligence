"""Retrieval over the enterprise corpus for RAG (EXP-07, EXP-08).

Corpus documents are split into overlapping chunks. Three retrievers:
- bm25: lexical, with crude Russian stemming, no extra dependencies;
- dense: Qwen3-Embedding (last-token pooling, L2-normalized);
- hybrid: reciprocal rank fusion of bm25 and dense.
Any of them can be followed by Qwen3-Reranker on the top candidates.

Build an index:
    python scripts/rag_retrieval.py --corpus data/extracted/corpus.jsonl --output-dir models/rag_index
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from generate_qa_dataset import detect_document_type, split_chunks

EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
RERANKER_MODEL = 'Qwen/Qwen3-Reranker-0.6B'
# The model cards recommend English task instructions.
RETRIEVAL_TASK = 'Given a question about an internal enterprise document, retrieve passages that answer the question'
DEFAULT_CHUNK_CHARS = 600
DEFAULT_OVERLAP_CHARS = 100
MODES = ('bm25', 'dense', 'hybrid')


# Chunking

def chunk_document(document, size=DEFAULT_CHUNK_CHARS, overlap=DEFAULT_OVERLAP_CHARS):
    text = (document.get('text') or '').strip()
    if not text:
        return []
    chunks = []
    previous = ''
    document_type = detect_document_type(text, document.get('relative_path'))
    for index, piece in enumerate(split_chunks(text, size)):
        tail = previous[-overlap:] if overlap and previous else ''
        if tail and ' ' in tail:
            tail = tail[tail.index(' ') + 1:]  # start the overlap at a word boundary
        chunks.append({
            'chunk_id': f"{document['document_id']}:{index}",
            'document_id': document['document_id'],
            'document_type': document_type,
            'file_name': document.get('file_name', ''),
            'text': f'{tail} {piece}'.strip() if tail else piece,
        })
        previous = piece
    return chunks


def chunk_corpus(documents, size=DEFAULT_CHUNK_CHARS, overlap=DEFAULT_OVERLAP_CHARS):
    return [chunk for document in documents for chunk in chunk_document(document, size, overlap)]


# BM25

def tokenize(text):
    words = re.sub(r'[^\w\s]', ' ', (text or '').lower().replace('ё', 'е')).split()
    # First 5 letters as a crude stem for Russian inflection; keep numbers whole.
    return [word if word.isdigit() else word[:5] for word in words if len(word) >= 3 or word.isdigit()]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.postings = {}
        for index, doc in enumerate(self.docs):
            for term in doc:
                self.postings.setdefault(term, []).append(index)

    def search(self, query, k=10):
        scores = Counter()
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings[term]:
                tf = self.docs[index][term]
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avg_length or 1))
                scores[index] += idf * tf * (self.k1 + 1) / norm
        return scores.most_common(k)


def reciprocal_rank_fusion(rankings, k=60, limit=None):
    """rankings: lists of chunk indices, best first. Returns [(index, score)] best first."""
    scores = Counter()
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] += 1.0 / (k + rank + 1)
    return scores.most_common(limit)


# Neural models (torch imported lazily so BM25 works without it)

class DenseEncoder:
    def __init__(self, model_name=EMBEDDING_MODEL, max_length=1024, batch_size=16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def encode(self, texts, is_query=False):
        import numpy as np

        torch = self.torch
        if is_query:
            texts = [f'Instruct: {RETRIEVAL_TASK}\nQuery:{text}' for text in texts]
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors='pt').to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            # Left padding: the last position is always a real token.
            pooled = torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.zeros((0, 0), dtype='float32')


class Reranker:
    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
              'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
              '<|im_start|>user\n')
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, model_name=RERANKER_MODEL, max_length=2048, batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.prefix_ids = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.max_length, self.batch_size = max_length, batch_size

    def score(self, query, texts):
        torch = self.torch
        pairs = [f'<Instruct>: {RETRIEVAL_TASK}\n<Query>: {query}\n<Document>: {text}' for text in texts]
        scores = []
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        for start in range(0, len(pairs), self.batch_size):
            enc = self.tokenizer(pairs[start:start + self.batch_size], padding=False, truncation='longest_first',
                                 return_attention_mask=False, max_length=budget)
            enc['input_ids'] = [self.prefix_ids + ids + self.suffix_ids for ids in enc['input_ids']]
            enc = self.tokenizer.pad(enc, padding=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            scores += torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return scores


# Index

def save_index(output_dir, chunks, embeddings=None, meta=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'chunks.jsonl').open('w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    if embeddings is not None:
        import numpy as np

        np.save(output_dir / 'embeddings.npy', embeddings.astype('float16'))
    (output_dir / 'meta.json').write_text(json.dumps(meta or {}, ensure_ascii=False, indent=2), encoding='utf-8')


class Retriever:
    """Search an index directory. encoder/reranker can be injected (tests) or loaded lazily."""

    def __init__(self, index_dir, encoder=None, reranker=None):
        index_dir = Path(index_dir)
        with (index_dir / 'chunks.jsonl').open('r', encoding='utf-8') as handle:
            self.chunks = [json.loads(line) for line in handle if line.strip()]
        self.meta = json.loads((index_dir / 'meta.json').read_text(encoding='utf-8'))
        self.bm25 = BM25([chunk['text'] for chunk in self.chunks])
        self.embeddings = None
        if (index_dir / 'embeddings.npy').exists():
            import numpy as np

            self.embeddings = np.load(index_dir / 'embeddings.npy').astype('float32')
        self._encoder, self._reranker = encoder, reranker

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = DenseEncoder(self.meta.get('embedding_model', EMBEDDING_MODEL))
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker(self.meta.get('reranker_model', RERANKER_MODEL))
        return self._reranker

    def _dense(self, query, k):
        if self.embeddings is None:
            raise RuntimeError('Index has no embeddings; rebuild without --no-dense.')
        scores = self.embeddings @ self.encoder.encode([query], is_query=True)[0]
        top = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def search(self, query, k=5, mode='hybrid', rerank=False, candidates=50):
        if mode not in MODES:
            raise ValueError(f'Unknown mode {mode!r}; expected one of {MODES}')
        pool = max(k, candidates) if rerank else k
        if mode == 'bm25':
            ranked = self.bm25.search(query, pool)
        elif mode == 'dense':
            ranked = self._dense(query, pool)
        else:
            lexical = [i for i, _ in self.bm25.search(query, max(pool, candidates))]
            dense = [i for i, _ in self._dense(query, max(pool, candidates))]
            ranked = reciprocal_rank_fusion([lexical, dense], limit=pool)
        if rerank and ranked:
            indices = [i for i, _ in ranked]
            scores = self.reranker.score(query, [self.chunks[i]['text'] for i in indices])
            ranked = sorted(zip(indices, scores), key=lambda pair: pair[1], reverse=True)
        return [{**self.chunks[i], 'score': round(float(s), 6)} for i, s in ranked[:k]]


def main():
    parser = argparse.ArgumentParser(description='Build a RAG index from corpus.jsonl.')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('models/rag_index'))
    parser.add_argument('--chunk-chars', type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument('--overlap-chars', type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument('--embedding-model', default=EMBEDDING_MODEL)
    parser.add_argument('--reranker-model', default=RERANKER_MODEL)
    parser.add_argument('--no-dense', action='store_true', help='BM25 only; no GPU or torch needed.')
    args = parser.parse_args()

    with args.corpus.open('r', encoding='utf-8') as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    chunks = chunk_corpus(documents, args.chunk_chars, args.overlap_chars)
    print(f'Documents: {len(documents)}, chunks: {len(chunks)}')

    embeddings = None
    if not args.no_dense:
        embeddings = DenseEncoder(args.embedding_model).encode([chunk['text'] for chunk in chunks])
    meta = {'corpus': str(args.corpus), 'documents': len(documents), 'chunks': len(chunks),
            'chunk_chars': args.chunk_chars, 'overlap_chars': args.overlap_chars,
            'embedding_model': None if args.no_dense else args.embedding_model,
            'reranker_model': args.reranker_model}
    save_index(args.output_dir, chunks, embeddings, meta)
    print(f'Index saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
