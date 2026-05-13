"""
FastAPI 入口 — RAG 企业知识库智能问答系统
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schema import HealthResponse

app = FastAPI(
    title="RAG Knowledge Base API",
    description="企业知识库智能问答系统 — 支持 RAG 检索增强生成",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 路由 ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


# TODO: Day 6 — 核心接口
# @app.post("/rag/query", response_model=QueryResponse)
# async def rag_query(req: QueryRequest):
#     ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
