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

## gstack

Use the `/browse` skill from gstack for all web browsing.
Never use `mcp__claude-in-chrome__*` tools.

### Available skills
| Skill | Description |
|---|---|
| `/office-hours` | Startup diagnostic + builder brainstorm |
| `/plan-ceo-review` | Strategy/scope review |
| `/plan-eng-review` | Architecture/engineering plan review |
| `/plan-design-review` | Design plan review (report-only) |
| `/design-consultation` | Design system from scratch |
| `/design-shotgun` | Visual design exploration |
| `/design-html` | HTML design generation |
| `/design-review` | Visual polish + fix loop |
| `/review` | PR code review |
| `/ship` | Ship workflow (merge + deploy) |
| `/land-and-deploy` | Merge → deploy → canary verify |
| `/canary` | Post-deploy monitoring loop |
| `/benchmark` | Performance regression detection |
| `/browse` | Headless browser (Playwright) |
| `/connect-chrome` | Launch GStack Browser |
| `/qa` | QA testing + fix loop |
| `/qa-only` | Report-only QA |
| `/setup-browser-cookies` | Browser cookie setup |
| `/setup-deploy` | One-time deploy config |
| `/setup-gbrain` | GBrain setup |
| `/retro` | Retrospective |
| `/investigate` | Systematic root-cause debugging |
| `/document-release` | Post-ship doc updates |
| `/document-generate` | Diataxis doc generator |
| `/codex` | Multi-AI second opinion |
| `/cso` | OWASP Top 10 + STRIDE security audit |
| `/autoplan` | Auto-review pipeline (CEO → design → eng) |
| `/plan-devex-review` | Developer experience review |
| `/devex-review` | Developer experience audit |
| `/careful` | Careful/safe mode |
| `/freeze` | Freeze mode |
| `/guard` | Guard mode |
| `/unfreeze` | Unfreeze mode |
| `/gstack-upgrade` | Upgrade gstack |
| `/learn` | Learning/tutorial mode |

### How teammates install gstack
1. Download `https://github.com/garrytan/gstack/archive/refs/heads/main.zip`
2. Extract to `%USERPROFILE%\.claude\skills\gstack` (strip the `-main` suffix)
3. Run in that directory:
   ```bash
   bun install && bun run build && bunx playwright install chromium
   ```
4. If on Windows without Git Bash, run these steps manually (see team setup guide).
