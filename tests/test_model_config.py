from pathlib import Path

import pytest

from generate_predictions import build_parser
from model_config import base_model_name, load_config, resolve_adapter

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ['configs/qlora.yaml', 'configs/qlora_mistral.yaml']


def test_main_config_is_qwen3():
    cfg = load_config(ROOT / 'configs' / 'qlora.yaml')
    assert cfg['model_name'] == 'Qwen/Qwen3-4B-Instruct-2507'
    assert cfg['output_dir'] == 'models/adapters/qwen3-4b-qlora'


def test_mistral_config_kept_for_comparison():
    cfg = load_config(ROOT / 'configs' / 'qlora_mistral.yaml')
    assert cfg['model_name'] == 'mistralai/Mistral-7B-Instruct-v0.3'
    assert cfg['output_dir'] != load_config(ROOT / 'configs' / 'qlora.yaml')['output_dir']


@pytest.mark.parametrize('path', CONFIGS)
def test_configs_have_same_training_setup(path):
    # Model comparison must only change the model, not the training recipe.
    cfg = load_config(ROOT / path)
    main = load_config(ROOT / 'configs' / 'qlora.yaml')
    for section in ('quantization', 'lora', 'training'):
        assert cfg[section] == main[section]


def test_base_model_override():
    cfg = {'model_name': 'Qwen/Qwen3-4B-Instruct-2507'}
    assert base_model_name(cfg) == 'Qwen/Qwen3-4B-Instruct-2507'
    assert base_model_name(cfg, 'other/model') == 'other/model'


def test_resolve_adapter():
    cfg = {'output_dir': 'models/adapters/qwen3-4b-qlora'}
    assert resolve_adapter(cfg) is None
    assert resolve_adapter(cfg, with_adapter=True) == 'models/adapters/qwen3-4b-qlora'
    assert resolve_adapter(cfg, 'custom/adapter', with_adapter=True) == 'custom/adapter'


def test_predictions_cli_defaults_to_config():
    args = build_parser().parse_args(['--input', 'test.jsonl', '--output', 'out.jsonl', '--with-adapter'])
    assert args.config == Path('configs/qlora.yaml')
    assert args.base_model is None
    assert args.with_adapter is True


def test_no_hardcoded_model_in_inference_code():
    for path in ('scripts/generate_predictions.py', 'inference/chat.py', 'training/train_qlora.py', 'Makefile'):
        source = (ROOT / path).read_text(encoding='utf-8')
        assert 'Mistral-7B' not in source, path
        assert 'Qwen3-4B-Instruct' not in source or path == 'Makefile', path


def test_merge_system_prompt_folds_into_user():
    from format_sft_dataset import SYSTEM_PROMPT, build_prompt_messages, merge_system_prompt

    messages = build_prompt_messages({'instruction': 'Кто?', 'input': 'Контекст'})
    merged = merge_system_prompt(messages)
    assert [m['role'] for m in merged] == ['user']
    assert merged[0]['content'].startswith(SYSTEM_PROMPT + '\n\nКто?')
    assert messages[0]['role'] == 'system'  # input not mutated
    assert merge_system_prompt(merged) == merged


def test_only_mistral_merges_system_prompt():
    assert load_config(ROOT / 'configs' / 'qlora_mistral.yaml').get('merge_system_prompt') is True
    assert not load_config(ROOT / 'configs' / 'qlora.yaml').get('merge_system_prompt')


def load_cached_tokenizer(name):
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('jinja2')
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f'{name} tokenizer is not cached locally')


@pytest.mark.parametrize('path', CONFIGS)
def test_training_prompt_keeps_system_and_fits(path):
    """Rendered training example contains the system prompt and fits max_seq_length."""
    from format_sft_dataset import SYSTEM_PROMPT, format_record, merge_system_prompt

    cfg = load_config(ROOT / path)
    tokenizer = load_cached_tokenizer(cfg['model_name'])
    record = {'document_id': 'd', 'instruction': 'Кто назначен ответственным за исполнение приказа?',
              'input': 'Ответственным за исполнение приказа назначить начальника отдела кадров. ' * 40,
              'output': 'Ответственным назначен начальник отдела кадров. ' * 3}
    item = format_record(record)
    prompt = merge_system_prompt(item['prompt']) if cfg.get('merge_system_prompt') else item['prompt']
    train_text = tokenizer.apply_chat_template(prompt + item['completion'], tokenize=False)
    infer_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    assert SYSTEM_PROMPT in train_text
    assert SYSTEM_PROMPT in infer_text
    assert '<think>' not in train_text
    assert len(tokenizer(train_text, add_special_tokens=False)['input_ids']) <= cfg['training']['max_seq_length']


def test_warmup_kwargs_supports_both_transformers_versions():
    from train_qlora import warmup_kwargs

    class Old:
        def __init__(self, warmup_ratio=0.0, warmup_steps=0):
            pass

    class New:
        def __init__(self, warmup_steps=0):
            pass

    assert warmup_kwargs(Old, 0.03) == {'warmup_ratio': 0.03}
    assert warmup_kwargs(New, 0.03) == {'warmup_steps': 0.03}
