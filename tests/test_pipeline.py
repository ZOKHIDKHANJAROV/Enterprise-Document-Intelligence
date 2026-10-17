import json
import subprocess
import sys
from pathlib import Path

import pytest

from format_sft_dataset import SYSTEM_PROMPT, build_prompt_messages, format_record, truncate_context
from generate_predictions import build_parser
from prepare_dataset import validate_jsonl
from train_qlora import fits_max_length

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def qa(document_id, index=0, **extra):
    return {'document_id': document_id, 'document_type': 'Приказ', 'instruction': f'Вопрос {index}?',
            'input': 'Контекст', 'output': f'Ответ {index}', **extra}


# 1. prepare_dataset keeps document_id, so the document-level split works.

def test_prepare_keeps_document_fields(tmp_path):
    src, dst = tmp_path / 'raw.jsonl', tmp_path / 'validated.jsonl'
    write_jsonl(src, [qa('doc-1'), {**qa(7), 'document_type': None}])
    assert validate_jsonl(src, dst) == 2
    rows = read_jsonl(dst)
    assert rows[0]['document_id'] == 'doc-1'
    assert rows[0]['document_type'] == 'Приказ'
    assert rows[1]['document_id'] == '7'
    assert rows[1]['document_type'] == 'unknown'


def test_prepare_rejects_missing_document_id(tmp_path):
    src = tmp_path / 'raw.jsonl'
    record = qa('doc-1')
    del record['document_id']
    write_jsonl(src, [record])
    with pytest.raises(ValueError, match='document_id'):
        validate_jsonl(src, tmp_path / 'out.jsonl')


def test_validate_then_split_has_no_document_leakage(tmp_path):
    raw, validated, out = tmp_path / 'raw.jsonl', tmp_path / 'validated.jsonl', tmp_path / 'split'
    write_jsonl(raw, [qa(f'doc-{d}', i) for d in range(40) for i in range(3)])
    validate_jsonl(raw, validated)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'split_dataset.py'), '--input', str(validated),
                    '--output-dir', str(out)], check=True, capture_output=True)

    splits = {name: read_jsonl(out / f'{name}.jsonl') for name in ('train', 'validation', 'test')}
    assert all(splits.values()), 'every split must be non-empty'
    assert sum(len(rows) for rows in splits.values()) == 120
    ids = {name: {r['document_id'] for r in rows} for name, rows in splits.items()}
    assert not ids['train'] & ids['validation']
    assert not ids['train'] & ids['test']
    assert not ids['validation'] & ids['test']


def test_split_fails_without_document_id(tmp_path):
    src = tmp_path / 'validated.jsonl'
    write_jsonl(src, [{'instruction': 'Вопрос?', 'input': '', 'output': 'Ответ'}])
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'split_dataset.py'), '--input', str(src),
                             '--output-dir', str(tmp_path / 'split')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'document_id' in result.stderr


# 2. Long contexts are truncated and over-length examples are dropped, so the
#    answer is never cut off by max_seq_length.

def test_truncate_context_limits_length():
    text = 'слово ' * 1000
    cut = truncate_context(text, 100)
    assert len(cut) <= 102
    assert cut.endswith('…')
    assert truncate_context('короткий текст', 100) == 'короткий текст'
    assert truncate_context(None, 100) == ''


def test_long_context_does_not_cut_answer():
    record = qa('doc-1', input='текст ' * 10000, output='Ответственный — начальник отдела.')
    item = format_record(record, max_context_chars=500)
    assert len(item['prompt'][1]['content']) < 600
    assert item['completion'][0]['content'] == 'Ответственный — начальник отдела.'


def test_fits_max_length_drops_long_examples():
    count_tokens = lambda example: len(example['text'].split())
    assert fits_max_length({'text': 'a ' * 10}, count_tokens, 10)
    assert not fits_max_length({'text': 'a ' * 11}, count_tokens, 10)


# 3. SFT data is prompt/completion, so the loss is computed only on the answer.

def test_format_record_is_prompt_completion():
    item = format_record(qa('doc-1'))
    assert 'messages' not in item
    assert [m['role'] for m in item['prompt']] == ['system', 'user']
    assert [m['role'] for m in item['completion']] == ['assistant']
    assert item['completion'][0]['content'] == 'Ответ 0'
    assert 'Ответ 0' not in item['prompt'][1]['content']
    assert item['document_id'] == 'doc-1'


def test_train_config_uses_completion_only_loss():
    source = (ROOT / 'training' / 'train_qlora.py').read_text(encoding='utf-8')
    assert 'completion_only_loss=True' in source
    assert "select_columns(['prompt', 'completion'])" in source


# 4. Predictions can run on the base model (EXP-01) with the training prompt.

def test_predictions_adapter_is_optional():
    args = build_parser().parse_args(['--input', 'test.jsonl', '--output', 'out.jsonl'])
    assert args.adapter is None
    args = build_parser().parse_args(['--input', 'test.jsonl', '--output', 'out.jsonl', '--adapter', 'models/a'])
    assert args.adapter == 'models/a'


def test_prediction_prompt_matches_training_prompt():
    record = qa('doc-1', input='текст ' * 1000)
    assert build_prompt_messages(record) == format_record(record)['prompt']
    assert build_prompt_messages(record)[0]['content'] == SYSTEM_PROMPT
