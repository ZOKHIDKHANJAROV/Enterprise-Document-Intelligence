import json
import subprocess
import sys
from pathlib import Path

import pytest

import judge_predictions
from add_unanswerable import UNANSWERABLE_ANSWER, is_refusal, make_unanswerable
from evaluate_generation import score_row, summarize
from judge_predictions import aggregate, build_judge_messages, parse_verdict

ROOT = Path(__file__).resolve().parents[1]

CONTEXTS = {
    'doc-1': 'Ответственным за исполнение приказа назначить начальника отдела кадров Иванову.',
    'doc-2': 'Поставщик обязуется доставить оборудование на склад заказчика в течение тридцати дней.',
    'doc-3': 'Совещание по бюджету переносится на понедельник, участники собираются в переговорной.',
}
QUESTIONS = {
    'doc-1': ('Кто ответственный за исполнение приказа?', 'Начальник отдела кадров Иванова'),
    'doc-2': ('Какой срок доставки оборудования?', 'Тридцать дней'),
    'doc-3': ('Когда совещание по бюджету?', 'В понедельник'),
}


def qa(document_id):
    question, answer = QUESTIONS[document_id]
    return {'document_id': document_id, 'document_type': 'Приказ', 'instruction': question,
            'input': CONTEXTS[document_id], 'output': answer}


# Unanswerable examples

def test_refusal_detection():
    assert is_refusal(UNANSWERABLE_ANSWER)
    assert is_refusal('В документе нет информации о сроках.')
    assert is_refusal('Контекст не содержит ответа на вопрос.')
    assert not is_refusal('Ответственный — начальник отдела кадров.')


def test_unanswerable_uses_other_document_context():
    records = [qa(d) for d in CONTEXTS]
    added = make_unanswerable(records, ratio=1.0, seed=1)
    assert len(added) == 3
    for example in added:
        assert example['answerable'] is False
        assert example['output'] == UNANSWERABLE_ANSWER
        assert example['document_id'] != example['question_document_id']
        assert example['input'] == CONTEXTS[example['document_id']]
        assert example['instruction'] == QUESTIONS[example['question_document_id']][0]


def test_unanswerable_rejects_overlapping_contexts():
    # Same wording in every document: any donor could contain the answer.
    records = [{**qa('doc-1'), 'document_id': f'doc-{i}'} for i in range(5)]
    assert make_unanswerable(records, ratio=1.0, seed=1) == []


def test_unanswerable_needs_two_documents():
    assert make_unanswerable([qa('doc-1')], ratio=1.0, seed=1) == []


def test_unanswerable_script_is_idempotent(tmp_path):
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps(qa(d), ensure_ascii=False) + '\n' for d in CONTEXTS), encoding='utf-8')
    command = [sys.executable, str(ROOT / 'scripts' / 'add_unanswerable.py'), '--dataset-dir', str(tmp_path), '--ratio', '1.0']
    for _ in range(2):
        subprocess.run(command, check=True, capture_output=True)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert sum(r.get('answerable', True) for r in rows) == 3
    assert sum(not r.get('answerable', True) for r in rows) == 3


# Refusal metrics

def test_summary_separates_answerable_and_unanswerable():
    rows = [score_row(r) for r in [
        {'answerable': True, 'reference': 'Иванова', 'prediction': 'Иванова'},
        {'answerable': True, 'reference': 'Тридцать дней', 'prediction': 'Недостаточно информации.'},
        {'answerable': False, 'reference': UNANSWERABLE_ANSWER, 'prediction': 'Нет информации об этом.'},
        {'answerable': False, 'reference': UNANSWERABLE_ANSWER, 'prediction': 'Генеральный директор.'},
    ]]
    summary = summarize(rows)
    assert summary['answerable_examples'] == 2
    assert summary['exact_match'] == 0.5
    assert summary['false_refusal_rate'] == 0.5
    assert summary['refusal_accuracy'] == 0.5


def test_summary_treats_old_predictions_as_answerable():
    summary = summarize([score_row({'reference': 'Иванова', 'prediction': 'Иванова'})])
    assert summary['answerable_examples'] == 1
    assert summary['exact_match'] == 1.0


# LLM judge

def test_judge_prompt_contains_all_parts():
    row = {'instruction': 'Кто?', 'context': 'Контекст', 'reference': 'Иванова', 'prediction': 'Петров'}
    user = build_judge_messages(row)[1]['content']
    for part in ('Кто?', 'Контекст', 'Иванова', 'Петров'):
        assert part in user


def test_parse_verdict_accepts_wrapped_json():
    verdict = parse_verdict('Вот оценка:\n```json\n{"correctness": "2", "faithful": true, "refusal": false, "comment": "ok"}\n```')
    assert verdict == {'correctness': 2, 'faithful': True, 'refusal': False, 'comment': 'ok'}


@pytest.mark.parametrize('text', ['нет json', '{"correctness": 3, "faithful": true, "refusal": false}',
                                  '{"correctness": 1, "faithful": "yes", "refusal": false}'])
def test_parse_verdict_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def verdict(correctness, faithful=True, refusal=False):
    return {'correctness': correctness, 'faithful': faithful, 'refusal': refusal, 'comment': ''}


def test_aggregate_metrics():
    rows = [
        {'answerable': True, 'judge': verdict(2)},
        {'answerable': True, 'judge': verdict(0, faithful=False)},
        {'answerable': True, 'judge': verdict(0, refusal=True)},
        {'answerable': False, 'judge': verdict(0, refusal=True)},
        {'answerable': False, 'judge': verdict(0, faithful=False)},
        {'answerable': True},  # judge error
    ]
    summary = aggregate(rows)
    assert summary['judged'] == 5
    assert summary['judge_errors'] == 1
    assert summary['correctness'] == round(2 / 6, 4)
    assert summary['fully_correct_rate'] == round(1 / 3, 4)
    assert summary['faithfulness'] == 0.5
    assert summary['false_refusal_rate'] == round(1 / 3, 4)
    assert summary['refusal_accuracy'] == 0.5
    # Unfaithful answer + non-refusal to unanswerable, out of 2 answered + 2 unanswerable.
    assert summary['hallucination_rate'] == 0.5


def test_judge_main_writes_summary(tmp_path, monkeypatch):
    predictions = tmp_path / 'predictions.jsonl'
    rows = [{'document_id': 'doc-1', 'answerable': True, 'instruction': 'Кто?', 'context': 'К',
             'reference': 'Иванова', 'prediction': 'Иванова'},
            {'document_id': 'doc-2', 'answerable': True, 'instruction': 'Что?', 'context': 'К',
             'reference': 'X', 'prediction': 'Y'}]
    predictions.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

    def fake_judge(base_url, model, row, timeout, api_key=None, no_think=False):
        if row['document_id'] == 'doc-2':
            raise ValueError('bad json')
        return verdict(2)

    monkeypatch.setattr(judge_predictions, 'judge', fake_judge)
    output = tmp_path / 'judge.json'
    monkeypatch.setattr('sys.argv', ['judge', '--predictions', str(predictions), '--output', str(output), '--model', 'm'])
    judge_predictions.main()
    summary = json.loads(output.read_text(encoding='utf-8'))['summary']
    assert summary['judged'] == 1
    assert summary['judge_errors'] == 1
    assert summary['correctness'] == 1.0
    assert summary['judge_model'] == 'm'


def test_judge_no_think_and_think_block():
    row = {'instruction': 'Q', 'context': 'C', 'reference': 'R', 'prediction': 'P'}
    assert build_judge_messages(row, no_think=True)[0]['content'].endswith('/no_think')
    assert '/no_think' not in build_judge_messages(row)[0]['content']
    verdict = parse_verdict('<think>\n\n</think>\n{"correctness": 1, "faithful": true, "refusal": false}')
    assert verdict['correctness'] == 1
