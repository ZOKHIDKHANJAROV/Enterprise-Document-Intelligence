import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = ('Ты корпоративный AI-ассистент. Отвечай только на основании '
                 'предоставленного контекста. Если контекста недостаточно, '
                 'прямо укажи, что информации недостаточно.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open('r', encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            user = record['instruction']
            if record.get('input'):
                user += '\n\nКонтекст документа:\n' + record['input']
            item = {'document_id': record['document_id'], 'document_type': record['document_type'], 'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': record['output']},
            ]}
            target.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'Formatted examples: {count}')


if __name__ == '__main__':
    main()
