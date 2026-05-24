# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

智能招聘助手 v2.0 — 求职 Agent 智能助手。FastAPI backend + FAISS vector search (Flat/HNSW/IVF/HNSW_PQ) + BM25 dual retrieval + BGE embeddings/reranker + DeepSeek LLM + LangGraph multi-agent。

38 个 API 端点，Gradio 6 页面前端，45 个测试。核心链路：PDF 解析 → AI 提取简历 → RAG 检索增强生成 → 投递自然语言管理 → 岗位智能推荐。

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
├── main.py       # FastAPI entry — 38 routes, CORS, WeChat webhook, startup init
├── config.py     # pydantic-settings — LLM/embedding/reranker/FAISS/chunk/HF mirror
├── api/          # 7 route modules: profile, rag, command, generate, applications, recommend
├── rag/          # Retrieval pipeline: embeddings, retriever(BM25+FAISS+persist), reranker, generator(stream+JSON-safe), pipeline
├── agent/        # Multi-agent: rag_agent(LangGraph 5-node), workflow(tracking+interview), scheduler(APScheduler)
├── data/         # Document: loader(PDF/Word/MD/TXT), pdf_parser(pymupdf+DeepSeek), chunker(5 strategies), jd_data
├── services/     # Business logic: extraction(general+LLM), generation, preparation, tracking
├── storage/      # Persistence: profile_store(JSON), application_store(SQLite), memory_store(JSON)
└── gateway/      # WeChat: wechat(adapter), router, push(placeholder)
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
