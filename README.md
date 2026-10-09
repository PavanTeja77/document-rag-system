# Document-Based Retrieval-Augmented Generation (RAG) System

An end-to-end RAG pipeline that answers questions from documents using hybrid retrieval, cross-encoder reranking, and a locally hosted large language model.

## Features

- **PDF Ingestion:** Extracts text and page metadata from PDF documents.
- **Text Chunking:** Splits documents into overlapping chunks for retrieval.
- **Semantic Search:** Uses Sentence Transformers embeddings and ChromaDB.
- **Keyword Search:** Uses BM25 for exact-term retrieval.
- **Hybrid Retrieval:** Combines dense and sparse rankings using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Reorders retrieved passages by query relevance.
- **Grounded Generation:** Uses retrieved context to generate answers with a local Qwen2.5-3B model through Ollama.
- **Source Citations:** Adds source-document and page references to answers.
- **Abstention Handling:** Returns a fallback response when the answer is not found in the provided context.

## Architecture

PDF Documents
↓
Text Extraction → Chunking and Metadata
↓
Dense Retrieval + BM25
↓
Reciprocal Rank Fusion
↓
Cross-Encoder Reranking
↓
Context Construction → Local LLM
↓
Answer with Source Citation

## Tech Stack

- **Language:** Python
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Keyword Retrieval:** BM25 (`rank-bm25`)
- **Rank Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Runtime:** Ollama
- **Language Model:** Qwen2.5-3B
- **PDF Processing:** PyMuPDF

## Project Structure

```text
RAG/
├── data/
│   └── papers/
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── bm25_store.py
│   ├── bm25_search.py
│   ├── hybrid_retrieval.py
│   ├── reranker.py
│   ├── context_builder.py
│   └── generator.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama from https://ollama.com/ and download the model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before generating answers.

### 5. Add a PDF document

Place your PDF in:

```text
data/papers/
```

Update the PDF path in the relevant source files if necessary.

### 6. Build the indexes

Run these commands from the project root, in order:

```bash
python src/vector_store.py
python src/bm25_store.py
```

### 7. Ask questions

```bash
python src/generator.py
```

Update the query in `generator.py` to ask a different question.

## Current Evaluation

The pipeline has been manually tested on factual, paraphrased, numerical, multi-hop, and unanswerable questions using a research-paper PDF.

These tests demonstrated end-to-end question answering, retrieval, reranking, source attribution, and fallback handling. A formal benchmark and quantitative retrieval evaluation remain future improvements.

## Future Improvements

- Structure-aware chunking and document cleaning
- Retrieval metrics such as Recall@K and MRR
- Automated evaluation datasets
- Better evidence-based citation selection
- Multi-document ingestion
- Streamlit interface

## Disclaimer

This project is intended for experimentation and learning. Generated answers should be verified against their cited source passages.
