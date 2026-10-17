import json

import numpy as np
import pytest

from evaluate_generation import score_row, summarize
from evaluate_retrieval import first_hit_rank, load_queries, retrieval_metrics
from rag_predict import CITATION_NOTE, build_rag_messages, format_sources, parse_citations
from rag_retrieval import BM25, Retriever, chunk_corpus, chunk_document, reciprocal_rank_fusion, save_index, tokenize

DOCS = [
    {'document_id': 'order', 'file_name': 'order.pdf', 'relative_path': 'Приказы/order.pdf',
     'text': 'ПРИКАЗ № 15\nОтветственным за исполнение назначить начальника отдела кадров Иванову.'},
    {'document_id': 'contract', 'file_name': 'contract.docx', 'relative_path': 'contract.docx',
     'text': 'ДОГОВОР поставки\nПоставщик доставляет оборудование на склад в течение 30 дней.'},
    {'document_id': 'memo', 'file_name': 'memo.docx', 'relative_path': 'memo.docx',
     'text': 'Служебная записка\nСовещание по бюджету переносится на понедельник.'},
]


# Chunking

def test_chunk_document_overlap_and_ids():
    text = ' '.join(f'Пункт {i} регламента о порядке работы.' for i in range(60))
    chunks = chunk_document({'document_id': 'd', 'text': text}, size=200, overlap=50)
    assert len(chunks) > 3
    assert [c['chunk_id'] for c in chunks[:2]] == ['d:0', 'd:1']
    assert all(len(c['text']) <= 200 + 50 for c in chunks)
    # Second chunk starts with the tail of the first one.
    assert chunks[1]['text'].split()[0] in chunks[0]['text']


def test_chunk_corpus_detects_type_and_skips_empty():
    chunks = chunk_corpus(DOCS + [{'document_id': 'empty', 'text': '  '}])
    assert {c['document_id'] for c in chunks} == {'order', 'contract', 'memo'}
    types = {c['document_id']: c['document_type'] for c in chunks}
    assert types == {'order': 'Приказ', 'contract': 'Договор', 'memo': 'Служебная записка'}


# BM25 and fusion

def test_tokenize_stems_and_keeps_numbers():
    assert tokenize('Приказом № 15 назначен') == ['прика', '15', 'назна']


def test_bm25_finds_inflected_words():
    bm25 = BM25([d['text'] for d in DOCS])
    top = bm25.search('кто отвечает за исполнение приказа', k=3)
    assert top[0][0] == 0
    assert bm25.search('срок доставки оборудования', k=1)[0][0] == 1
    assert bm25.search('абракадабра', k=3) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], limit=3)
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[2][0] in (3, 4)


# Retriever with injected models

class FakeEncoder:
    """Embeds by keyword so the test controls which chunk is 'semantically' closest."""
    KEYS = ['ответствен', 'оборудован', 'бюджет']

    def encode(self, texts, is_query=False):
        vectors = np.array([[float(k in t.lower()) for k in self.KEYS] for t in texts]) + 1e-3
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeReranker:
    def __init__(self, preferred):
        self.preferred = preferred

    def score(self, query, texts):
        return [1.0 if self.preferred in t else 0.0 for t in texts]


@pytest.fixture
def index_dir(tmp_path):
    chunks = chunk_corpus(DOCS)
    save_index(tmp_path, chunks, FakeEncoder().encode([c['text'] for c in chunks]), {'chunks': len(chunks)})
    return tmp_path


@pytest.mark.parametrize('mode', ['bm25', 'dense', 'hybrid'])
def test_retriever_modes_find_source(index_dir, mode):
    retriever = Retriever(index_dir, encoder=FakeEncoder())
    results = retriever.search('Кто ответственный за исполнение?', k=2, mode=mode)
    assert results[0]['document_id'] == 'order'
    assert 'score' in results[0]


def test_retriever_rerank_reorders(index_dir):
    retriever = Retriever(index_dir, encoder=FakeEncoder(), reranker=FakeReranker('Совещание'))
    results = retriever.search('Кто ответственный за исполнение?', k=1, mode='dense', rerank=True, candidates=3)
    assert results[0]['document_id'] == 'memo'


def test_retriever_without_embeddings(tmp_path):
    save_index(tmp_path, chunk_corpus(DOCS), None, {})
    retriever = Retriever(tmp_path)
    assert retriever.search('бюджет', k=1, mode='bm25')[0]['document_id'] == 'memo'
    with pytest.raises(RuntimeError, match='no embeddings'):
        retriever.search('бюджет', mode='dense')
    with pytest.raises(ValueError):
        retriever.search('бюджет', mode='unknown')


# Retrieval evaluation

def test_retrieval_metrics():
    chunks = [{'document_id': 'a'}, {'document_id': 'b'}, {'document_id': 'b'}]
    assert first_hit_rank(chunks, 'b') == 2
    assert first_hit_rank(chunks, 'z') is None
    metrics = retrieval_metrics([1, 2, None, 6], ks=(1, 5))
    assert metrics == {'recall@1': 0.25, 'recall@5': 0.5, 'mrr': round((1 + 0.5 + 1 / 6) / 4, 4), 'queries': 4}


def test_load_queries_skips_unanswerable_and_duplicates(tmp_path):
    path = tmp_path / 'test.jsonl'
    rows = [{'document_id': 'a', 'instruction': 'Q1', 'output': 'x'},
            {'document_id': 'a', 'instruction': 'Q1', 'output': 'x'},
            {'document_id': 'b', 'instruction': 'Q2', 'output': 'y', 'answerable': False}]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    assert [q['instruction'] for q in load_queries(path)] == ['Q1']


# RAG prompt and citations

CHUNKS = [{'document_id': 'order', 'file_name': 'order.pdf', 'text': 'Ответственный — Иванова.'},
          {'document_id': 'memo', 'file_name': '', 'text': 'Совещание в понедельник.'},
          {'document_id': 'order', 'file_name': 'order.pdf', 'text': 'Срок — март.'}]


def test_format_sources_numbers_chunks():
    text = format_sources(CHUNKS)
    assert text.startswith('[1] order.pdf\nОтветственный — Иванова.')
    assert '[2] memo\n' in text
    assert '[3] order.pdf' in text


def test_rag_messages_keep_all_sources_and_citation_note():
    long_chunks = [{**CHUNKS[0], 'text': 'слово ' * 500}] * 5
    messages = build_rag_messages('Кто ответственный?', long_chunks)
    assert [m['role'] for m in messages] == ['system', 'user']
    assert CITATION_NOTE in messages[1]['content']
    assert messages[1]['content'].count('[5] order.pdf') == 1  # not truncated
    assert [m['role'] for m in build_rag_messages('Q', CHUNKS, merge_system=True)] == ['user']


def test_parse_citations():
    assert parse_citations('Иванова [1], совещание [2][3].', CHUNKS) == ['order', 'memo']
    assert parse_citations('См. [2, 1] и [9].', CHUNKS) == ['memo', 'order']
    assert parse_citations('Без ссылок.', CHUNKS) == []


def test_rag_citation_metrics():
    rows = [score_row(r) for r in [
        {'document_id': 'order', 'reference': 'Иванова', 'prediction': 'Иванова [1]',
         'retrieved_document_ids': ['order', 'memo'], 'cited_document_ids': ['order']},
        {'document_id': 'contract', 'reference': '30 дней', 'prediction': 'Не знаю',
         'retrieved_document_ids': ['memo'], 'cited_document_ids': []},
    ]]
    summary = summarize(rows)
    assert summary['retrieval_hit_rate'] == 0.5
    assert summary['citation_rate'] == 0.5
    assert summary['citation_accuracy'] == 0.5
    assert 'citation_rate' not in summarize([score_row({'reference': 'a', 'prediction': 'a'})])
