import json
from collections import Counter

import generate_qa_dataset as qa
from generate_qa_dataset import (build_examples, detect_document_type, grounding_score, load_existing,
                                 select_context, split_chunks)

ORDER_TEXT = ('ООО «Ромашка»\nПРИКАЗ № 15\nот 01.03.2026\n'
              'Об утверждении Положения о командировках\n\n'
              'Ответственным за исполнение назначить начальника отдела кадров Иванову А.А.\n'
              'Срок исполнения — до 31 марта 2026 года.')


def document(text=ORDER_TEXT, document_id='doc-1', relative_path='orders/order.pdf'):
    return {'document_id': document_id, 'file_name': 'order.pdf', 'relative_path': relative_path, 'text': text}


# Document type

def test_detects_type_from_heading_earliest_match():
    assert detect_document_type(ORDER_TEXT) == 'Приказ'


def test_detects_longer_type_names_first():
    assert detect_document_type('ДОЛЖНОСТНАЯ ИНСТРУКЦИЯ бухгалтера') == 'Должностная инструкция'
    assert detect_document_type('Служебная записка о закупке') == 'Служебная записка'


def test_act_does_not_match_inside_words():
    assert detect_document_type('Актуальный список активов') == 'unknown'
    assert detect_document_type('АКТ приёмки выполненных работ') == 'Акт'


def test_type_falls_back_to_folder_then_unknown():
    assert detect_document_type('Без заголовка', 'Договоры/2026/file.pdf') == 'Договор'
    assert detect_document_type('Без заголовка', 'misc/file.pdf') == 'unknown'
    assert detect_document_type('', None) == 'unknown'


# Grounding and context selection

def test_grounding_score_handles_word_forms():
    assert grounding_score('Начальник отдела кадров Иванова', ORDER_TEXT) == 1.0
    assert grounding_score('Генеральный директор Петров', ORDER_TEXT) == 0.0


def test_select_context_uses_real_fragment_only():
    real = 'Срок исполнения — до 31 марта 2026 года.'
    assert select_context(ORDER_TEXT, real, 'Какой срок?', 'До 31 марта') == real
    fake = 'Срок исполнения — до 1 января 2030 года.'
    assert select_context(ORDER_TEXT, fake, 'Какой срок?', 'До 31 марта') == ORDER_TEXT


def test_select_context_picks_chunk_with_answer_in_long_document():
    filler = '\n'.join(f'Пункт {i}. Общие положения о порядке работы подразделения.' for i in range(200))
    text = filler + '\nОтветственным назначить главного инженера Сидорова.\n' + filler
    context = select_context(text, '', 'Кто ответственный?', 'Главный инженер Сидоров', size=500)
    assert len(context) <= 500
    assert 'Сидорова' in context


def test_split_chunks_respects_size():
    chunks = split_chunks('а' * 1200 + '\n' + 'б' * 100, 500)
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert ''.join(chunks).replace('\n', '') == 'а' * 1200 + 'б' * 100


# Building examples: dedupe, grounding filter, type

def test_build_examples_filters_and_dedupes():
    raw = [
        {'instruction': 'Кто ответственный?', 'input': '', 'output': 'Начальник отдела кадров Иванова А.А.'},
        {'instruction': 'Кто  ответственный?!', 'input': '', 'output': 'Иванова'},
        {'instruction': 'Кто подписал?', 'input': '', 'output': 'Генеральный директор Петров'},
        {'instruction': '', 'input': '', 'output': 'пусто'},
        'not a dict',
    ]
    stats = Counter()
    records = build_examples(document(), raw, 1800, 0.6, stats)
    assert [r['instruction'] for r in records] == ['Кто ответственный?']
    assert records[0]['document_type'] == 'Приказ'
    assert records[0]['document_id'] == 'doc-1'
    assert records[0]['grounding'] == 1.0
    assert stats == {'duplicates': 1, 'ungrounded': 1, 'empty': 1}


# Resume

def test_resume_skips_processed_documents(tmp_path, monkeypatch):
    corpus, output = tmp_path / 'corpus.jsonl', tmp_path / 'qa.jsonl'
    docs = [document(document_id='doc-1'), document(document_id='doc-2')]
    corpus.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in docs), encoding='utf-8')
    output.write_text(json.dumps({'document_id': 'doc-1', 'instruction': 'Старый вопрос?', 'output': 'x'},
                                 ensure_ascii=False) + '\n', encoding='utf-8')
    calls = []

    def fake_generate(base_url, model, text, count, timeout):
        calls.append(text)
        return [{'instruction': 'Какой срок?', 'input': '', 'output': 'До 31 марта 2026 года'}]

    monkeypatch.setattr(qa, 'generate', fake_generate)
    monkeypatch.setattr('sys.argv', ['qa', '--input', str(corpus), '--output', str(output), '--model', 'm', '--resume'])
    qa.main()

    assert len(calls) == 1
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert [r['document_id'] for r in rows] == ['doc-1', 'doc-2']
    assert set(load_existing(output)) == {'doc-1', 'doc-2'}
