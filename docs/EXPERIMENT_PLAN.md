# Experiment plan

## EXP-00 Dataset sanity

Goal: verify that generated QA examples are grounded in source documents.

Checks:
- no empty instruction or answer;
- source document exists;
- duplicate question rate;
- document-level split integrity;
- average input/output length;
- document-type distribution.

## EXP-01 Base model baseline

Run the base Qwen/Qwen3-4B-Instruct-2507 model on the held-out test set without the adapter.
Record generation settings, latency and evaluation metrics.

## EXP-02 QLoRA

Train the same base model with the same train/validation split.
Only the LoRA adapter is trainable.

Initial parameters:
- r = 16
- alpha = 32
- dropout = 0.05
- target_modules = all-linear
- 4-bit NF4
- double quantization
- learning rate = 1e-4
- sequence length = 1024
- batch size = 1
- gradient accumulation = 8
- epochs = 2

## EXP-03 Dataset scaling

Compare approximately 500, 2,000 and 7,557 source documents.
Keep the evaluation set fixed where possible.

## EXP-04 LoRA rank

Compare r = 8, 16 and 32 while keeping all other variables fixed.

## EXP-05 Context length

Compare 1024 and 2048 tokens after establishing a stable 1024-token run.

## EXP-06 Model comparison

Repeat the controlled experiment with mistralai/Mistral-7B-Instruct-v0.3
(configs/qlora_mistral.yaml, RUN=mistral-7b). On an 8 GB GPU the 7B model may
need max_seq_length 768. Any change to the comparison checkpoint is recorded
in the experiment configuration.

## EXP-07 RAG integration

Use the QLoRA adapter as the generation model and add retrieval over the full
enterprise corpus.

## Metrics

Training:
- train loss;
- validation loss;
- tokens processed;
- training time.

Quality:
- answer correctness;
- faithfulness;
- context relevance;
- hallucination rate;
- lexical baseline metrics.

Systems:
- peak VRAM;
- generation latency;
- tokens/second;
- adapter size.

## Reproducibility

Every experiment must record:
- git commit SHA;
- model identifier;
- dataset version/hash;
- configuration file;
- random seed;
- GPU model;
- CUDA/PyTorch/Transformers/TRL/PEFT versions.
