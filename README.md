# Enterprise Document Intelligence

Enterprise Document Intelligence is a research project for adapting an instruction-tuned LLM to enterprise document workflows with QLoRA, followed by RAG integration.

## Project roadmap

1. Repository bootstrap and reproducible environment
2. Document ingestion and corpus audit
3. Instruction/QA dataset construction
4. QLoRA fine-tuning of Mistral 7B
5. Evaluation against the base model
6. RAG integration
7. Hybrid retrieval, reranking, citations, and production API

## Initial experiment

Target model: `mistralai/Mistral-7B-Instruct-v0.3`

Planned training setup:
- 4-bit NF4 quantization
- double quantization
- LoRA adapters
- gradient checkpointing
- 8-bit paged AdamW
- sequence length starting at 1024
- batch size 1 with gradient accumulation

The first training run will use a small verified subset of the enterprise corpus before scaling to the full document collection.

## Repository structure

```text
enterprise-document-intelligence/
├── data/
│   ├── raw/
│   ├── extracted/
│   └── dataset/
├── scripts/
├── training/
├── inference/
├── evaluation/
├── models/
├── configs/
├── notebooks/
└── requirements.txt
```

Document files and model weights are intentionally not committed to Git.
