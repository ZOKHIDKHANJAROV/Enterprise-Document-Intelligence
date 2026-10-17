# LLM judge endpoint; override with make judge JUDGE_MODEL=... JUDGE_URL=...
JUDGE_URL ?= http://localhost:8000/v1
JUDGE_MODEL ?= judge

.PHONY: gpu extract inspect qa qa-resume validate split unanswerable report format train chat predict-base evaluate-base judge-base predict evaluate judge

gpu:
	python scripts/check_gpu.py

extract:
	python scripts/extract_documents.py --input-dir data/raw/documents --output data/extracted/corpus.jsonl --max-documents 500 --ocr

inspect:
	python scripts/inspect_corpus.py --input data/extracted/corpus.jsonl

qa:
	python scripts/generate_qa_dataset.py --input data/extracted/corpus.jsonl --output data/dataset/qa_raw.jsonl --base-url http://localhost:8000/v1 --model mistral --max-documents 500

qa-resume:
	python scripts/generate_qa_dataset.py --input data/extracted/corpus.jsonl --output data/dataset/qa_raw.jsonl --base-url http://localhost:8000/v1 --model mistral --max-documents 500 --resume

validate:
	python scripts/prepare_dataset.py --input data/dataset/qa_raw.jsonl --output data/dataset/qa_validated.jsonl

split:
	python scripts/split_dataset.py --input data/dataset/qa_validated.jsonl --output-dir data/dataset

unanswerable:
	python scripts/add_unanswerable.py --dataset-dir data/dataset --ratio 0.1

report:
	python scripts/dataset_report.py --input data/dataset/all.jsonl --corpus data/extracted/corpus.jsonl --split-dir data/dataset --strict

format:
	python scripts/format_sft_dataset.py --input data/dataset/train.jsonl --output data/dataset/train_sft.jsonl
	python scripts/format_sft_dataset.py --input data/dataset/validation.jsonl --output data/dataset/validation_sft.jsonl

train:
	python training/train_qlora.py --train data/dataset/train_sft.jsonl --validation data/dataset/validation_sft.jsonl

chat:
	python inference/chat.py --adapter models/adapters/mistral-qlora

predict-base:
	python scripts/generate_predictions.py --input data/dataset/test.jsonl --output evaluation/predictions_base.jsonl

evaluate-base:
	python scripts/evaluate_generation.py --predictions evaluation/predictions_base.jsonl --output evaluation/results_base.json

judge-base:
	python scripts/judge_predictions.py --predictions evaluation/predictions_base.jsonl --output evaluation/judge_base.json --base-url $(JUDGE_URL) --model $(JUDGE_MODEL)

predict:
	python scripts/generate_predictions.py --input data/dataset/test.jsonl --output evaluation/predictions.jsonl --adapter models/adapters/mistral-qlora

evaluate:
	python scripts/evaluate_generation.py --predictions evaluation/predictions.jsonl --output evaluation/results.json

judge:
	python scripts/judge_predictions.py --predictions evaluation/predictions.jsonl --output evaluation/judge.json --base-url $(JUDGE_URL) --model $(JUDGE_MODEL)
