import json
import subprocess
import sys
from pathlib import Path

import yaml

from dataset_report import build_report, review_sample
from prepare_dataset import validate_jsonl

ROOT = Path(__file__).resolve().parents[1]


def qa(document_id, question='Кто отвечает?', **extra):
    return {'document_id': document_id, 'document_type': 'Приказ', 'instruction': question,
            'input': 'Контекст документа', 'output': 'Начальник отдела', **extra}


def test_clean_dataset_has_no_problems():
    records = [qa('doc-1', grounding=1.0), qa('doc-2', 'Какой срок?', grounding=0.7)]
    splits = {'train': [records[0]], 'validation': [records[1]], 'test': [qa('doc-3')]}
    report, problems = build_report(records, splits, corpus_ids={'doc-1', 'doc-2', 'doc-3'})
    assert problems == []
    assert report['examples'] == 2
    assert report['documents'] == 2
    assert report['document_types'] == {'Приказ': 2}
    assert report['grounding']['below_0.8'] == 1
    assert report['missing_source_documents'] == 0


def test_reports_duplicates_empty_and_missing_sources():
    records = [qa('doc-1'), qa('doc-1', 'Кто  отвечает?!'), qa('doc-2'), qa('doc-9', output='')]
    report, problems = build_report(records, corpus_ids={'doc-1', 'doc-2'})
    assert report['duplicate_questions_within_document'] == 1
    assert report['duplicate_question_rate'] == 0.75  # same question in 4 records
    assert report['empty_records'] == 1
    assert report['missing_source_documents'] == 1
    assert any('empty' in p for p in problems)
    assert any('not found in corpus' in p for p in problems)


def test_detects_split_leakage_and_empty_split():
    splits = {'train': [qa('doc-1')], 'validation': [qa('doc-1')], 'test': []}
    _, problems = build_report(splits['train'] + splits['validation'], splits)
    assert any('both train and validation' in p for p in problems)
    assert any('test split is empty' in p for p in problems)


def test_review_sample_is_markdown_with_checkboxes():
    text = review_sample([qa('doc-1', grounding=0.9)], 10, seed=1)
    assert text.startswith('# QA review sample')
    assert '**Вопрос:** Кто отвечает?' in text
    assert '**Grounding:** 0.9' in text
    assert '- [ ] верно' in text


def test_strict_mode_fails_on_problems(tmp_path):
    data = tmp_path / 'all.jsonl'
    data.write_text(json.dumps(qa('doc-1', output=''), ensure_ascii=False) + '\n', encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'dataset_report.py'), '--input', str(data),
                             '--output', str(tmp_path / 'r.json'), '--sample-output', str(tmp_path / 's.md'),
                             '--strict'], capture_output=True)
    assert result.returncode == 1
    assert json.loads((tmp_path / 'r.json').read_text(encoding='utf-8'))['empty_records'] == 1


def test_prepare_keeps_grounding_score(tmp_path):
    src, dst = tmp_path / 'raw.jsonl', tmp_path / 'out.jsonl'
    src.write_text(json.dumps(qa('doc-1', grounding=0.75), ensure_ascii=False) + '\n', encoding='utf-8')
    validate_jsonl(src, dst)
    assert json.loads(dst.read_text(encoding='utf-8'))['grounding'] == 0.75


def test_qlora_config_uses_bf16_consistently():
    cfg = yaml.safe_load((ROOT / 'configs' / 'qlora.yaml').read_text(encoding='utf-8'))
    assert cfg['quantization']['compute_dtype'] == 'bfloat16'
    assert cfg['training']['bf16'] is True
    assert cfg['training']['fp16'] is False


def test_chat_uses_training_system_prompt():
    source = (ROOT / 'inference' / 'chat.py').read_text(encoding='utf-8')
    assert 'from format_sft_dataset import SYSTEM_PROMPT' in source
    assert "'content': SYSTEM_PROMPT" in source


def test_prepare_drops_noisy_pairs_and_joins_list_answers(tmp_path):
    src, dst = tmp_path / 'raw.jsonl', tmp_path / 'out.jsonl'
    rows = [qa('doc-1'),
            qa('doc-2', input='Мехжш,t,нн ,ttухофilЗt} ýýjIHш хспраёнила и{100нки}лг'),
            qa('doc-3', output="['С 01.07.74', 'до 01.07.94']")]
    src.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    stats = {}
    assert validate_jsonl(src, dst, max_noise=0.3, stats=stats) == 2
    assert stats == {'noisy': 1}
    out = [json.loads(line) for line in dst.read_text(encoding='utf-8').splitlines()]
    assert [r['document_id'] for r in out] == ['doc-1', 'doc-3']
    assert out[1]['output'] == 'С 01.07.74; до 01.07.94'
