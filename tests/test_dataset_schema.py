import json
from pathlib import Path


def test_sft_record_shape(tmp_path: Path):
    record = {
        'document_id': 'doc-1',
        'document_type': 'Приказ',
        'instruction': 'Кто отвечает?',
        'input': 'Контекст',
        'output': 'Ответ',
    }
    path = tmp_path / 'sample.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    loaded = json.loads(path.read_text(encoding='utf-8').strip())
    assert {'document_id', 'document_type', 'instruction', 'input', 'output'} <= set(loaded)
    assert loaded['instruction']
    assert loaded['output']
