# Efficient RAG with LLMs for Medical Question Answering

## Overview
This repository contains the codebase for a research project that systematically evaluates Retrieval-Augmented Generation (RAG) for medical question answering under realistic single-GPU constraints. The study compares sparse, dense, and hybrid retrieval methods, alongside diagnostic baselines that test retrieval relevance, context alignment, and knowledge-base composition. Experiments are conducted across multiple medical QA datasets using open-source 7B–8B class language models, with a focus on both accuracy and computational efficiency.

## Highlights

- Comprehensive evaluation across MedMCQA, MedQA-USMLE, and PubMedQA

- Supports three instruction/chat-tuned LLMs:

  - LLaMA-3-8B-Instruct

  - Mistral-7B-Instruct

  - DeepSeek-7B-Chat

- Implements:

  - Sparse retrieval (BM25)

  - Dense retrieval (GTR-T5, S-BioBERT)

  - Hybrid fusion + reranking

  - Diagnostic baselines (Random retrieval, Dataset-context, Dataset-context-injection)

- Joint evaluation of effectiveness (Accuracy, Macro-F1) and efficiency (latency, input length)

- Fully reproducible pipeline using INT8 quantization to support single-GPU execution

## Datasets

This project evaluates three widely-used medical QA benchmarks:

| Dataset         | Description                                                                   | Citation         |
| --------------- | ----------------------------------------------------------------------------- | ---------------- |
| **MedMCQA**     | 194k multi-subject MCQs; 1,001-question validation split used in this project | Pal et al., 2022 |
| **MedQA-USMLE** | 1,273 clinical vignette questions requiring multi-step reasoning              | Jin et al., 2021 |
| **PubMedQA**    | 1,000 research-focused yes/no/maybe questions grounded in PubMed abstracts    | Jin et al., 2019 |

Dataset contexts for PubMedQA were also used for oracle and sensitivity-based retrieval experiments.

## Knowledge Base Construction
Three corpus conditions are used:
 1. External Knowledge Base
    - 23 medical sources, including textbooks, clinical manuals, and open-access journals
    - Approximately 14 million words (119 MB) processed using LangChain + Tesseract OCR
    - Chunking: 500-character segments with 50-character overlap
Additional corpora used for PubMedQA experiments:
 2. Dataset Context Only (PubMedQA):
    - Compact, highly aligned abstracts provided by the dataset (approx. 212k words)
 3. Merged Knowledge Base
    - External KB + dataset contexts (approximately 14.3M words)

## Retrieval Methods
Implemented retrieval approaches:
1. Sparse Retrieval
   - BM25 (Pyserini) over inverted indices
   - High lexical precision; low computational cost

2. Dense Retrieval
   - FAISS exact inner-product search
   - Encoders: GTR-T5-Large, S-BioBERT
   - No ANN approximation used

3. Hybrid Retrieval
   - Linear fusion of sparse + dense scores
   - Cross-encoder reranking (MiniLM/monoT5)
   - Returns top-5 final passages

4. Diagnostic Baselines
   - Random retrieval: noise baseline for relevance
   - Dataset-context retrieval (PubMedQA): oracle baseline
   - Dataset-context-injection ((merged knowledge base)): sensitivity test for large-corpus retrieval

All retrieval methods return five passages per query to ensure controlled comparison.

## LLMs and Inference Setup
Three instruction/chat-tuned models are evaluated:
- LLaMA-3-8B-Instruct
- Mistral-7B-Instruct
- DeepSeek-7B-Chat
All models use:
- INT8 weight-only quantization (Dettmers et al., 2022)
- FP16 activations
- Greedy decoding (no sampling)
Hardware:
- NVIDIA RTX 4070 SUPER (12 GB VRAM)
- Intel i5-14600KF CPU, 32 GB RAM

## Evaluation
Effectiveness:
- Accuracy
- Macro-F1

Efficiency:
- End-to-end latency per query
- Average input word length (prefill proxy)
Statistical tests (Wilcoxon signed-rank) applied where sample size allows.

Logged CSVs include:
- Ground truth
- Model prediction
- Latency
- Input length

## Reproducibility
The pipeline is fully deterministic:
- Fixed random seeds
- Greedy decoding
- Pinned library versions
- No approximations in FAISS retrieval

The full codebase, configurations, and logs are included for full reproduction of all 81 experimental configurations.

