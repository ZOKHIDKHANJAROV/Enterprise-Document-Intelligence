# Model under test. EXP-06 Mistral run: make train predict evaluate CONFIG=configs/qlora_mistral.yaml RUN=mistral-7b
CONFIG ?= configs/qlora.yaml
RUN ?= qwen3-4b
EVAL_DIR = evaluation/$(RUN)

# Source documents, one folder per document type.
DOCS_DIR ?= data/raw/documents
# Sampling for the controlled experiment; quality filter drops broken text layers.
EXTRACT_FLAGS ?= --min-text-chars 300 --min-quality 0.55

# OpenAI-compatible endpoint used to generate QA pairs. Default: Ollama with
# qwen3-4b-qa (see README: qwen3:4b-instruct with num_ctx 8192).
QA_URL ?= http://localhost:11434/v1
QA_MODEL ?= qwen3-4b-qa

# RAG: index over INDEX_CORPUS (use corpus_full.jsonl from make extract-full for the whole corpus).
INDEX_CORPUS ?= data/extracted/corpus.jsonl
INDEX_DIR ?= models/rag_index
RAG_FLAGS ?= --mode hybrid --rerank --top-k 5

# LLM judge endpoint; override with make judge JUDGE_MODEL=... JUDGE_URL=...
JUDGE_URL ?= http://localhost:8000/v1
JUDGE_MODEL ?= judge

.PHONY: gpu extract inspect qa qa-resume validate split unanswerable report format extract-full index index-bm25 eval-retrieval rag-base rag rag-evaluate-base rag-evaluate train chat predict-base evaluate-base judge-base predict evaluate judge

gpu:
	python scripts/check_gpu.py

extract:
	python scripts/extract_documents.py --input-dir $(DOCS_DIR) --output data/extracted/corpus.jsonl --max-documents 500 --max-per-folder 120 $(EXTRACT_FLAGS)

inspect:
	python scripts/inspect_corpus.py --input data/extracted/corpus.jsonl

qa:
	python scripts/generate_qa_dataset.py --input data/extracted/corpus.jsonl --output data/dataset/qa_raw.jsonl --base-url $(QA_URL) --model $(QA_MODEL) --max-documents 500

qa-resume:
	python scripts/generate_qa_dataset.py --input data/extracted/corpus.jsonl --output data/dataset/qa_raw.jsonl --base-url $(QA_URL) --model $(QA_MODEL) --max-documents 500 --resume

validate:
	python scripts/prepare_dataset.py --input data/dataset/qa_raw.jsonl --output data/dataset/qa_validated.jsonl --max-noise 0.3

split:
	python scripts/split_dataset.py --input data/dataset/qa_validated.jsonl --output-dir data/dataset --train-ratio 0.8 --validation-ratio 0.1

unanswerable:
	python scripts/add_unanswerable.py --dataset-dir data/dataset --ratio 0.1

report:
	python scripts/dataset_report.py --input data/dataset/all.jsonl --corpus data/extracted/corpus.jsonl --split-dir data/dataset --strict

format:
	python scripts/format_sft_dataset.py --input data/dataset/train.jsonl --output data/dataset/train_sft.jsonl
	python scripts/format_sft_dataset.py --input data/dataset/validation.jsonl --output data/dataset/validation_sft.jsonl

train:
	python training/train_qlora.py --config $(CONFIG) --train data/dataset/train_sft.jsonl --validation data/dataset/validation_sft.jsonl $(TRAIN_FLAGS)

chat:
	python inference/chat.py --config $(CONFIG) --with-adapter

predict-base:
	python scripts/generate_predictions.py --input data/dataset/test.jsonl --output $(EVAL_DIR)/predictions_base.jsonl --config $(CONFIG)

evaluate-base:
	python scripts/evaluate_generation.py --predictions $(EVAL_DIR)/predictions_base.jsonl --output $(EVAL_DIR)/results_base.json

judge-base:
	python scripts/judge_predictions.py --predictions $(EVAL_DIR)/predictions_base.jsonl --output $(EVAL_DIR)/judge_base.json --base-url $(JUDGE_URL) --model $(JUDGE_MODEL)

predict:
	python scripts/generate_predictions.py --input data/dataset/test.jsonl --output $(EVAL_DIR)/predictions.jsonl --config $(CONFIG) --with-adapter

evaluate:
	python scripts/evaluate_generation.py --predictions $(EVAL_DIR)/predictions.jsonl --output $(EVAL_DIR)/results.json

judge:
	python scripts/judge_predictions.py --predictions $(EVAL_DIR)/predictions.jsonl --output $(EVAL_DIR)/judge.json --base-url $(JUDGE_URL) --model $(JUDGE_MODEL)

extract-full:
	python scripts/extract_documents.py --input-dir $(DOCS_DIR) --output data/extracted/corpus_full.jsonl $(EXTRACT_FLAGS)

index:
	python scripts/rag_retrieval.py --corpus $(INDEX_CORPUS) --output-dir $(INDEX_DIR)

index-bm25:
	python scripts/rag_retrieval.py --corpus $(INDEX_CORPUS) --output-dir $(INDEX_DIR) --no-dense

eval-retrieval:
	python scripts/evaluate_retrieval.py --input data/dataset/test.jsonl --index-dir $(INDEX_DIR) --output $(EVAL_DIR)/retrieval.json --rerank

rag-base:
	python scripts/rag_predict.py --input data/dataset/test.jsonl --output $(EVAL_DIR)/rag_base.jsonl --index-dir $(INDEX_DIR) --config $(CONFIG) $(RAG_FLAGS)

rag:
	python scripts/rag_predict.py --input data/dataset/test.jsonl --output $(EVAL_DIR)/rag.jsonl --index-dir $(INDEX_DIR) --config $(CONFIG) --with-adapter $(RAG_FLAGS)

rag-evaluate-base:
	python scripts/evaluate_generation.py --predictions $(EVAL_DIR)/rag_base.jsonl --output $(EVAL_DIR)/rag_results_base.json

rag-evaluate:
	python scripts/evaluate_generation.py --predictions $(EVAL_DIR)/rag.jsonl --output $(EVAL_DIR)/rag_results.json
