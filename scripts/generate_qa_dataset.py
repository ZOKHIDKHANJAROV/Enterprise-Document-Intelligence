import argparse
import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

SYSTEM_PROMPT = '''Ты готовишь QA-датасет для корпоративного AI-ассистента.
Используй только факты из документа. Не выдумывай. Создай вопросы разных
типов: факты, обязанности, сроки, ограничения, процедуры и краткое содержание.
Не начинай вопрос с названия типа (не пиши "Факт:", "Процедура:" и т. п.).
Пиши вопрос и ответ на том же языке, что и документ (русском или узбекском),
не переводи. Числа, даты, номера и названия копируй из документа дословно.
Верни только JSON-массив объектов с полями instruction, input, output.
В поле input дословно скопируй фрагмент документа (2–5 предложений), на
котором основан ответ.'''

# Fragments shorter than this (e.g. just a document number) are not useful
# training context; the surrounding document chunk is used instead.
MIN_FRAGMENT_CHARS = 200
CATEGORY_PREFIX = re.compile(
    r'^(?:факт\w*|обязанност\w*|срок\w*|ограничени\w*|процедур\w*|краткое содержание|резюме|summary)\s*[:.\-–—]\s*',
    re.IGNORECASE)

# Matches the context budget used by format_sft_dataset.py.
DEFAULT_CONTEXT_CHARS = 1800
DEFAULT_MIN_GROUNDING = 0.6

# Longer names go first so "Должностная инструкция" wins over "Инструкция".
DOCUMENT_TYPES = [
    ('Должностная инструкция', r'должностн\w*\s+инструкц'),
    ('Служебная записка', r'служебн\w*\s+записк'),
    ('Докладная записка', r'докладн\w*\s+записк'),
    ('Пояснительная записка', r'пояснительн\w*\s+записк'),
    ('Штатное расписание', r'штатн\w*\s+расписани'),
    ('Организационно-технические мероприятия', r'организационно[-\s]+техническ'),
    ('Нормативный документ', r'нормативн'),
    ('Приказ', r'приказ'),
    ('Распоряжение', r'распоряжени'),
    ('Постановление', r'постановлени'),
    ('Решение', r'решени'),
    ('Положение', r'положени'),
    ('Инструкция', r'инструкци'),
    ('Регламент', r'регламент'),
    ('Договор', r'договор'),
    ('Соглашение', r'соглашени'),
    ('Протокол', r'протокол'),
    ('Акт', r'акт(?:а|е|у|ом|ы)?(?!\w)'),
    ('Отчёт', r'отч[её]т'),
    ('Справка', r'справк'),
    ('Заявление', r'заявлени'),
    ('Доверенность', r'доверенност'),
    ('Письмо', r'письм'),
    ('Калькуляция', r'калькуляци'),
    ('Смета', r'смет[аыеу]?(?!\w)'),
    ('Лимит', r'лимит'),
    ('Лицензия', r'лицензи'),
    ('План', r'план(?:ы|а|е)?(?!\w)'),
    ('Показатели', r'показател'),
    ('Политика', r'политик'),
    ('Правила', r'правил[аео]?(?!\w)'),
    ('Стандарт', r'стандарт'),
    ('Рапорт', r'рапорт'),
    ('Проект', r'проект'),
]
TYPE_HEAD_CHARS = 600
# Draft stamps ("ПРОЕКТ") precede the real type in headings; trust only folders.
FOLDER_ONLY_TYPES = {'Проект'}


def detect_document_type(text, relative_path=None):
    """Type from the folder name (e.g. "Приказы/..."), else from the heading, else 'unknown'.

    Folders are curated by type, so they win. In the heading the earliest type
    word wins, so "ПРИКАЗ ... Об утверждении Положения" is a Приказ.
    """
    if relative_path:
        for part in reversed(Path(relative_path).parts[:-1]):
            for name, pattern in DOCUMENT_TYPES:
                if re.match(pattern, part.lower()):
                    return name
    head = (text or '')[:TYPE_HEAD_CHARS].lower()
    best = None
    for name, pattern in DOCUMENT_TYPES:
        if name in FOLDER_ONLY_TYPES:
            continue
        match = re.search(r'(?<!\w)' + pattern, head)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), name)
    return best[1] if best else 'unknown'


def normalize(text):
    text = re.sub(r'[^\w\s]', ' ', (text or '').lower().replace('ё', 'е'))
    return re.sub(r'\s+', ' ', text).strip()


def stems(text):
    # Crude stemming for Russian morphology: compare the first 5 letters of
    # content words and skip short function words.
    return {word[:5] for word in normalize(text).split() if len(word) >= 4}


def grounding_score(answer, context):
    """Share of the answer's content words that occur in the context."""
    answer_stems = stems(answer)
    if not answer_stems:
        return 1.0
    return len(answer_stems & stems(context)) / len(answer_stems)


def split_chunks(text, size):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
    chunks, current = [], ''
    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ''
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        if current and len(current) + 1 + len(paragraph) > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f'{current}\n{paragraph}' if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def select_context(document_text, fragment, question, answer, size=DEFAULT_CONTEXT_CHARS):
    """Context for a QA pair that fits the training budget and contains the answer.

    Uses the model's quoted fragment when it really occurs in the document,
    otherwise the document chunk that best matches the question and answer.
    """
    fragment = (fragment or '').strip()
    real_fragment = bool(fragment) and normalize(fragment) in normalize(document_text)
    if real_fragment and MIN_FRAGMENT_CHARS <= len(fragment) <= size:
        return fragment
    if len(document_text) <= size:
        return document_text
    target = stems(question) | stems(answer) | (stems(fragment) if real_fragment else set())
    return max(split_chunks(document_text, size), key=lambda chunk: len(target & stems(chunk)))


def load_existing(path):
    """document_id -> normalized questions already written, for --resume."""
    done = {}
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                done.setdefault(record['document_id'], set()).add(normalize(record['instruction']))
    return done


def build_examples(document, raw_examples, context_chars, min_grounding, stats):
    """Turn model output into dataset records, dropping ungrounded and duplicate pairs."""
    text = document.get('text', '').strip()
    document_type = detect_document_type(text, document.get('relative_path'))
    seen = set()
    records = []
    for example in raw_examples:
        if not isinstance(example, dict):
            continue
        instruction = CATEGORY_PREFIX.sub('', str(example.get('instruction', '')).strip())
        output = str(example.get('output', '')).strip()
        if not instruction or not output:
            stats['empty'] += 1
            continue
        key = normalize(instruction)
        if key in seen:
            stats['duplicates'] += 1
            continue
        context = select_context(text, str(example.get('input', '')), instruction, output, context_chars)
        score = grounding_score(output, context)
        if score < min_grounding:
            stats['ungrounded'] += 1
            continue
        seen.add(key)
        records.append({'document_id': document['document_id'], 'document_type': document_type,
                        'instruction': instruction, 'input': context, 'output': output,
                        'grounding': round(score, 3)})
    return records


def parse_json(text):
    # Reasoning models (Qwen3 hybrid) may prepend a <think> block.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end <= start:
        raise ValueError('No JSON array in model response')
    payload = text[start:end + 1]
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        # Models copy backslashes from documents ("\e", "\G"); escape invalid ones.
        result = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', payload))
    if not isinstance(result, list):
        raise ValueError('Expected JSON array')
    return result


def generate(base_url, model, document_text, count, timeout, retries=2):
    """On a context-size error (dense tables tokenize poorly) retry with half the text."""
    for attempt in range(retries + 1):
        prompt = f'Создай {count} QA-примеров для документа.\n\nDOCUMENT:\n{document_text}'
        response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers={'Content-Type': 'application/json'}, json={'model': model, 'temperature': 0.2, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]}, timeout=timeout)
        if response.status_code == 400 and 'context' in response.text and attempt < retries:
            document_text = document_text[:len(document_text) // 2]
            continue
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
    parser.add_argument('--max-chars', type=int, default=10000)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--context-chars', type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument('--min-grounding', type=float, default=DEFAULT_MIN_GROUNDING,
                        help='Minimum share of answer content words found in the context.')
    parser.add_argument('--resume', action='store_true', help='Append to --output and skip documents already in it.')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open('r', encoding='utf-8') as source:
        documents = [json.loads(line) for line in source if line.strip()][:args.max_documents]

    done = load_existing(args.output) if args.resume else {}
    if done:
        print(f'Resuming: {len(done)} documents already in {args.output}')
    stats = {'documents': 0, 'skipped_done': 0, 'skipped_empty': 0, 'failed': 0,
             'examples': 0, 'empty': 0, 'duplicates': 0, 'ungrounded': 0}

    with args.output.open('a' if args.resume else 'w', encoding='utf-8') as target:
        for document in tqdm(documents, desc='Generating QA'):
            if document['document_id'] in done:
                stats['skipped_done'] += 1
                continue
            text = document.get('text', '').strip()
            if not text:
                stats['skipped_empty'] += 1
                continue
            try:
                raw = generate(args.base_url, args.model, text[:args.max_chars], args.questions_per_document, args.timeout)
            except Exception as exc:
                stats['failed'] += 1
                print(f"WARNING: {document.get('file_name')}: {exc}")
                continue
            stats['documents'] += 1
            for record in build_examples(document, raw, args.context_chars, args.min_grounding, stats):
                target.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['examples'] += 1
            target.flush()

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
