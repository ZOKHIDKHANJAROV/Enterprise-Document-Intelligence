.PHONY: gpu extract qa validate split format train chat predict evaluate

gpu:
	python scripts/check_gpu.py

extract:
	python scripts/extract_documents.py --input-dir data/raw/documents --output data/extracted/corpus.jsonl --max-documents 500

qa:
	python scripts/generate_qa_dataset.py --input data/extracted/corpus.jsonl --output data/dataset/qa_raw.jsonl --base-url http://localhost:8000/v1 --model mistral --max-documents 500

validate:
	python scripts/prepare_dataset.py --input data/dataset/qa_raw.jsonl --output data/dataset/qa_validated.jsonl

split:
	python scripts/split_dataset.py --input data/dataset/qa_validated.jsonl --output-dir data/dataset

format:
	python scripts/format_sft_dataset.py --input data/dataset/train.jsonl --output data/dataset/train_sft.jsonl
	python scripts/format_sft_dataset.py --input data/dataset/validation.jsonl --output data/dataset/validation_sft.jsonl

train:
	python training/train_qlora.py --train data/dataset/train_sft.jsonl --validation data/dataset/validation_sft.jsonl

chat:
	python inference/chat.py --adapter models/adapters/mistral-qlora

predict:
	python scripts/generate_predictions.py --input data/dataset/test.jsonl --output evaluation/predictions.jsonl --adapter models/adapters/mistral-qlora

evaluate:
	python scripts/evaluate_generation.py --predictions evaluation/predictions.jsonl --output evaluation/results.json
