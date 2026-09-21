import argparse
import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

SYSTEM_PROMPT = '''Ты готовишь QA-датасет для корпоративного AI-ассистента.
Используй только факты из документа. Не выдумывай. Создай вопросы разных
типов: факты, обязанности, сроки, ограничения, процедуры и краткое содержание.
Верни только JSON-массив объектов с полями instruction, input, output.'''


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end <= start:
        raise ValueError('No JSON array in model response')
    result = json.loads(text[start:end + 1])
    if not isinstance(result, list):
        raise ValueError('Expected JSON array')
    return result


def generate(base_url, model, document_text, count, timeout):
    prompt = f'Создай {count} QA-примеров для документа.\n\nDOCUMENT:\n{document_text}'
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers={'Content-Type': 'application/json'}, json={'model': model, 'temperature': 0.2, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]}, timeout=timeout)
    response.raise_for_status()
    return parse_json(response.json()['choices'][0]['message']['content'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--questions-per-document', type=int, default=5)
    parser.add_argument('--max-documents', type=int, default=500)
    parser.add_argument('--max-chars', type=int, default=24000)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open('r', encoding='utf-8') as source:
        documents = [json.loads(line) for line in source if line.strip()][:args.max_documents]

    with args.output.open('w', encoding='utf-8') as target:
        for document in tqdm(documents, desc='Generating QA'):
            text = document.get('text', '').strip()[:args.max_chars]
            if not text:
                continue
            try:
                examples = generate(args.base_url, args.model, text, args.questions_per_document, args.timeout)
            except Exception as exc:
                print(f"WARNING: {document.get('file_name')}: {exc}")
                continue
            for example in examples:
                if not isinstance(example, dict):
                    continue
                instruction = str(example.get('instruction', '')).strip()
                output = str(example.get('output', '')).strip()
                input_text = str(example.get('input', text)).strip()
                if not instruction or not output:
                    continue
                target.write(json.dumps({'document_id': document['document_id'], 'document_type': document.get('extension', 'unknown'), 'instruction': instruction, 'input': input_text, 'output': output}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
