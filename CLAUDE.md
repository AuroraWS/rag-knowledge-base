# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RAG-based enterprise knowledge base Q&A system. FastAPI backend + FAISS vector search + BGE embeddings/reranker + DeepSeek LLM + multi-agent collaboration.

The project is in early skeleton stage — core routes are stubbed out and modules have docstrings describing planned content but no implementation yet.

## Commands

```bash
# Create environment (conda, recommended)
conda env create -f environment.yml
conda activate rag-knowledge-base

# Or: venv + pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure (edit .env after copying)
cp .env.example .env

# Pre-download embedding model (one-time)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# Run dev server
uvicorn app.main:app --reload

# Run tests
pytest
```

## Architecture

```
app/
├── main.py       # FastAPI entry — routes: /health (done), /rag/query (stubbed)
├── config.py     # pydantic-settings from .env — DEEPSEEK_API_KEY, embedding/reranker models, top_k, dirs
├── rag/          # Retrieval pipeline (embeddings, BM25+vector dual retrieval, reranker, generator, pipeline orchestration)
├── agent/        # Multi-agent system (LangGraph/AutoGen workflow)
└── data/         # Document ingestion: loader (PDF/Word/MD), chunker, FAISS indexer
```

## Configuration (.env)

`app/config.py` uses `pydantic-settings` with auto-load from `.env`. Key vars: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `HOST`, `PORT`, `TOP_K`, `RERANK_TOP_K`, `KNOWLEDGE_BASE_DIR`, `FAISS_INDEX_DIR`.

## Tech stack

| Concern | Choice |
|----------|--------|
| LLM API | openai SDK (DeepSeek, OpenAI-compatible) |
| Embedding | BAAI/bge-small-zh-v1.5 via sentence-transformers |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Vector DB | FAISS (HNSW-PQ) |
| Sparse retrieval | BM25 (rank-bm25) |
| Document parsing | python-docx, pymupdf, markdown |
| Demo UI | Gradio |
| Tests | pytest + pytest-asyncio |

## Model storage

Models cache to the default `sentence-transformers` cache dir (~/.cache/torch/sentence_transformers). FAISS indexes and source documents are excluded from git (`.gitignore`).
