# SFT dataset

Training data is intentionally excluded from Git.

Expected source record:

```json
{
  "document_id": "unique-document-id",
  "document_type": "Приказ",
  "instruction": "Кто отвечает за выполнение приказа?",
  "input": "Relevant document text...",
  "output": "Согласно документу, ответственным является ..."
}
```

The dataset is split by `document_id`, not by individual examples. This prevents
examples generated from the same document from leaking into validation or test.

Generated files:

- `all.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
