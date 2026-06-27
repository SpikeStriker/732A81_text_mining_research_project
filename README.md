# 732A81_text_mining_research_project

This repository combines two related research tracks: Indian mythology data preparation and a systematic evaluation of retrieval-augmented generation (RAG) architectures for financial question answering.

## Repository structure

Top-level folders in this repository:

- `indian_mythology`
- `EvaluatingRagArchitecture`

## What is included

### indian_mythology

- Dataset engineering notebooks, pre-annotated JSONL corpora, and data analysis workflows for mythology-related NLP research.
- Includes Dockerized processing steps and prepared data files for experiment reproduction.

### EvaluatingRagArchitecture

- A full RAG experimentation pipeline for comparing multiple retrieval and generation architectures on the FiQA financial QA dataset.
- The report in `EvaluatingRagArchitecture/Report/report.pdf` documents the study design, experimental setup, and results.
- The notebooks in `EvaluatingRagArchitecture/notebooks` cover data download, Qdrant indexing, environment setup, and architecture evaluation.

## What the RAG study investigated

The project compares five RAG architectures:

- Naive RAG
- Re-Ranking RAG
- Hypothetical Document Embeddings (HyDE)
- Query Expansion RAG
- Multi-Step Iterative RAG

The experiments use the FiQA dataset with 57,638 documents and 30 evaluation QA pairs, and they assess lexical, semantic, fluency, and retrieval quality metrics.

## Key results from the report

- Re-Ranking RAG achieved the strongest lexical performance, with roughly 8% higher ROUGE-L and 177% higher BLEU than Naive RAG.
- Naive RAG preserved semantics more consistently, with about 5% higher cosine similarity and lower variability across runs.
- Retrieval quality showed weak correlation with final answer quality, with correlations below about 0.30, suggesting that stronger retrieval scores do not always translate into better generated answers.
- The study also found that a smaller set of metrics may be sufficient for future evaluation: ROUGE-L for lexical quality, cosine similarity and BERTScore F1 for semantic quality, BLEU for fluency, and retrieval similarity for retrieval quality.

## How the notebooks work

- `Download Data.ipynb` downloads the FiQA dataset and writes the corpus, evaluation, test, train, and validation JSONL files into the data folder.
- `EvaluateRagArchitectures.ipynb` loads the evaluation set, builds a Qdrant collection from the corpus, runs the RAG architectures, computes metrics, and saves experiment results as pickle files.
- `test Setup.ipynb` verifies the Python environment, PyTorch/CUDA availability, and embedding model dependencies.
- `uploadToQdrant.ipynb` uploads the corpus documents into Qdrant for retrieval-based experiments.

## How to use this repo

Browse the two main folders to explore the research notebook workflow, the report, the prepared datasets, and the evaluation assets.

## How to contribute

Open issues for corrections or improvements, and open pull requests with clear descriptions when extending the analysis or experiment pipeline.
