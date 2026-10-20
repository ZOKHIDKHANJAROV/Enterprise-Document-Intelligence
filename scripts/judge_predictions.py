"""Score predictions with an LLM judge (OpenAI-compatible chat API).

For each prediction the judge sees the question, the document context, the
reference answer and the model answer, and returns correctness (0-2),
faithfulness to the context and whether the answer is a refusal. Use a judge
model stronger than, or at least different from, the evaluated model.
"""

import argparse
import json
import os
import re
from pathlib import Path

import requests
from tqdm import tqdm

JUDGE_PROMPT = '''Ты строгий эксперт, проверяющий ответы корпоративного AI-ассистента.
Тебе дают вопрос, контекст документа, эталонный ответ и ответ модели.

Оцени ответ модели:
- correctness: 2 — по смыслу совпадает с эталоном; 1 — частично верен или неполон; 0 — неверен.
- faithful: true, если все утверждения ответа подтверждаются контекстом; false, если есть факты, которых нет в контексте или которые ему противоречат. Отказ отвечать считается faithful.
- refusal: true, если модель говорит, что в контексте недостаточно информации, и не даёт ответа по существу.

Верни только JSON-объект: {"correctness": 0|1|2, "faithful": true|false, "refusal": true|false, "comment": "кратко почему"}'''


def build_judge_messages(row, no_think=False):
    user = (f"ВОПРОС:\n{row['instruction']}\n\n"
            f"КОНТЕКСТ:\n{row.get('context') or '(нет)'}\n\n"
            f"ЭТАЛОННЫЙ ОТВЕТ:\n{row['reference']}\n\n"
            f"ОТВЕТ МОДЕЛИ:\n{row['prediction']}")
    # /no_think switches off reasoning in hybrid Qwen3 models used as judges.
    system = JUDGE_PROMPT + ('\n/no_think' if no_think else '')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_verdict(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('No JSON object in judge response')
    data = json.loads(text[start:end + 1])
    correctness = data.get('correctness')
    if isinstance(correctness, str) and correctness.strip().isdigit():
        correctness = int(correctness)
    if correctness not in (0, 1, 2):
        raise ValueError(f'Invalid correctness: {correctness!r}')
    for field in ('faithful', 'refusal'):
        if not isinstance(data.get(field), bool):
            raise ValueError(f'Invalid {field}: {data.get(field)!r}')
    return {'correctness': correctness, 'faithful': data['faithful'], 'refusal': data['refusal'],
            'comment': str(data.get('comment', ''))[:500]}


def rate(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate(rows):
    judged = [r for r in rows if 'judge' in r]
    answerable = [r['judge'] for r in judged if r.get('answerable', True)]
    unanswerable = [r['judge'] for r in judged if not r.get('answerable', True)]
    answered = [v for v in answerable if not v['refusal']]
    # A hallucination is an unfaithful answer to an answerable question, or
    # any non-refusal to an unanswerable one.
    hallucinations = sum(not v['faithful'] for v in answered) + sum(not v['refusal'] for v in unanswerable)
    return {
        'examples': len(rows),
        'judged': len(judged),
        'judge_errors': len(rows) - len(judged),
        'correctness': rate([v['correctness'] / 2 for v in answerable]),
        'fully_correct_rate': rate([v['correctness'] == 2 for v in answerable]),
        'faithfulness': rate([v['faithful'] for v in answered]),
        'false_refusal_rate': rate([v['refusal'] for v in answerable]),
        'refusal_accuracy': rate([v['refusal'] for v in unanswerable]),
        'hallucination_rate': rate([1] * hallucinations + [0] * (len(answered) + len(unanswerable) - hallucinations)),
    }


def judge(base_url, model, row, timeout, api_key=None, no_think=False):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers,
                             json={'model': model, 'temperature': 0, 'messages': build_judge_messages(row, no_think)},
                             timeout=timeout)
    response.raise_for_status()
    return parse_verdict(response.json()['choices'][0]['message']['content'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('evaluation/judge_results.json'))
    parser.add_argument('--base-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-env', default=None, help='Name of the environment variable with the API key, if needed.')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--no-think', action='store_true', help='Disable reasoning of hybrid Qwen3 judges.')
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    with args.predictions.open('r', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if rows and 'context' not in rows[0]:
        print('WARNING: predictions have no context; re-run generate_predictions.py so faithfulness can be judged.')

    for row in tqdm(rows, desc='Judging'):
        try:
            row['judge'] = judge(args.base_url, args.model, row, args.timeout, api_key, args.no_think)
        except Exception as exc:
            print(f"WARNING: {row.get('document_id')}: {re.sub(r'\s+', ' ', str(exc))[:200]}")

    summary = {**aggregate(rows), 'judge_model': args.model}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'examples': rows}, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
