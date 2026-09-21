# Enterprise Document Intelligence

Research project for adapting an instruction-tuned LLM to enterprise document
workflows with QLoRA, followed by RAG integration.

The target corpus is approximately 7,557 enterprise documents.

## Research question

How much does domain-specific QLoRA improve an instruction-tuned LLM on
enterprise document tasks, and how does the adapter behave when later combined
with RAG?

## Architecture

    Enterprise documents
            |
            v
    Document extraction
            |
            v
    Clean corpus JSONL
            |
            v
    QA generation
            |
            v
    QA validation
            |
            v
    Document-level train/validation/test split
            |
            v
    Mistral 7B + 4-bit NF4
            |
            v
    QLoRA adapter
            |
            +----------------------+
            |                      |
            v                      v
    Base vs QLoRA evaluation     Later: RAG
                                      |
                                      v
                            QLoRA + retrieval + citations

## Current implementation

Data pipeline:
- recursive PDF, DOCX, XLSX, TXT and MD extraction;
- SHA-256 document identity;
- basic text cleaning;
- corpus statistics;
- QA dataset validation;
- document-level dataset splitting;
- conversational SFT formatting.

QLoRA pipeline:
- Mistral-7B-Instruct-v0.3;
- 4-bit NF4 quantization;
- double quantization;
- PEFT LoRA;
- all-linear target modules;
- gradient checkpointing;
- paged 8-bit AdamW;
- FP16 compute;
- initial sequence length 1024;
- batch size 1 with gradient accumulation.

Hugging Face documents QLoRA as 4-bit quantization combined with trainable
LoRA weights and recommends NF4 for 4-bit training. The current PEFT
documentation also supports all-linear targeting for QLoRA-style training.

TRL supports standard and conversational SFT datasets and PEFT adapters. The
project therefore keeps SFT data in structured messages format.

The selected Mistral checkpoint is the official
mistralai/Mistral-7B-Instruct-v0.3 model.

## Repository structure

    configs/
        data.yaml
        qlora.yaml
    data/
        raw/
        extracted/
        dataset/
    scripts/
        check_gpu.py
        extract_documents.py
        generate_qa_dataset.py
        generate_predictions.py
        prepare_dataset.py
        split_dataset.py
        format_sft_dataset.py
        evaluate_generation.py
    training/
        train_qlora.py
    inference/
        chat.py
    evaluation/
    models/
    notebooks/
    Makefile
    pyproject.toml
    requirements.txt

## Dataset contract

Each QA record contains:

    {
      "document_id": "stable-document-id",
      "document_type": "Приказ",
      "instruction": "Кто отвечает за выполнение приказа?",
      "input": "Relevant document context...",
      "output": "Ответ только по документу..."
    }

The split is performed by document_id rather than by individual QA rows.
This prevents examples generated from the same document from appearing in
both training and test data.

## First controlled experiment

Do not start with all 7,557 documents.

First experiment:

    500 documents
        |
        v
    approximately 2,500 QA examples
        |
        v
    document-level split
        |
        v
    QLoRA
        |
        v
    held-out evaluation

After the pipeline is reviewed, scale to the complete corpus.

## Experiments

EXP-01  Base Mistral baseline
EXP-02  Mistral + QLoRA
EXP-03  Dataset-size scaling
EXP-04  LoRA rank comparison
EXP-05  Sequence-length comparison
EXP-06  Mistral vs Llama
EXP-07  QLoRA + RAG
EXP-08  Hybrid retrieval + reranking

Planned metrics:
- validation loss;
- answer correctness;
- faithfulness;
- hallucination rate;
- lexical baseline metrics;
- latency;
- VRAM usage;
- training time;
- adapter size.

## Workflow

    make gpu

    make extract

    make qa

    make validate

    make split

    make format

    make train

    make predict

    make evaluate

The commands are documented but have not been executed during repository
preparation.

## Data and security

- Do not commit enterprise documents.
- Do not commit model weights or LoRA checkpoints.
- Do not commit Hugging Face tokens or API keys.
- Review generated QA examples before training.
- Keep the test set isolated.
