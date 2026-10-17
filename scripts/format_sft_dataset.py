import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = ('Ты корпоративный AI-ассистент. Отвечай только на основании '
                 'предоставленного контекста. Если контекста недостаточно, '
                 'прямо укажи, что информации недостаточно.')

# Keeps system prompt + question + context + answer within max_seq_length=1024
# tokens for Cyrillic text. Over-length examples are still dropped at training.
DEFAULT_MAX_CONTEXT_CHARS = 1800


def truncate_context(text, max_chars):
    text = (text or '').strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer to cut at a whitespace boundary instead of mid-word.
    space = cut.rfind(' ')
    if space > max_chars * 0.8:
        cut = cut[:space]
    return cut.rstrip() + ' …'


def build_prompt_messages(record, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    """System + user messages shared by training and prediction."""
    user = record['instruction']
    context = truncate_context(record.get('input'), max_context_chars)
    if context:
        user += '\n\nКонтекст документа:\n' + context
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]


def format_record(record, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    # Conversational prompt-completion format: TRL computes the loss only on
    # the completion (assistant answer), not on the system prompt and context.
    return {
        'document_id': record['document_id'],
        'document_type': record.get('document_type', 'unknown'),
        'prompt': build_prompt_messages(record, max_context_chars),
        'completion': [{'role': 'assistant', 'content': record['output']}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-context-chars', type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open('r', encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for line in source:
            if not line.strip():
                continue
            item = format_record(json.loads(line), args.max_context_chars)
            target.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'Formatted examples: {count}')


if __name__ == '__main__':
    main()
