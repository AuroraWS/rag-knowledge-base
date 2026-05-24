"""RAG 知识库 API — 文档导入 / 向量检索 / 流式 LLM 生成

路由前缀: /api/rag

端点:
- POST /api/rag/documents/import  上传文档 → 解析 → 分块 → 嵌入 → 建索引
- POST /api/rag/search            向量+BM25双路检索（不经过LLM）
- POST /api/rag/query              完整RAG：检索→重排序→生成
- POST /api/rag/query/stream       流式SSE RAG查询
- GET  /api/rag/index/status       索引统计
- POST /api/rag/index/rebuild      从文档目录重建索引
- POST /api/rag/agent/query        多Agent RAG查询（Phase 4集成）
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.data.chunker import chunk_documents
from app.data.loader import _LOADERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG 知识库"])

# ── 单例 ───────────────────────────────────────────

_pipeline: Any = None  # RAGPipeline


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.rag.pipeline import RAGPipeline

        _pipeline = RAGPipeline()
        # 尝试从磁盘恢复之前持久化的索引（避免重启后重建）
        if not _pipeline.retriever.try_load_from_disk():
            logger.info("未找到持久化索引，需通过 POST /api/rag/documents/import 导入文档")
    return _pipeline


# ── 请求/响应模型 ──────────────────────────────────


class RagSearchRequest(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量")


class RagSearchResult(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float


class RagSearchResponse(BaseModel):
    results: list[RagSearchResult]
    latency_ms: float


class RagQueryRequest(BaseModel):
    model_config = {"extra": "forbid"}
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=50)
    rerank_top_k: int = Field(default=3, ge=1, le=20)
    system_prompt: str = Field(default="", description="可选的系统提示词")


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float


class IndexStatusResponse(BaseModel):
    document_count: int
    index_type: str
    dimension: int


# ── 端点 ───────────────────────────────────────────


@router.post("/documents/import", summary="导入文档")
async def import_documents(
    files: list[UploadFile] = File(..., description="支持 PDF/DOCX/MD/TXT 格式"),
    chunk_size: int = Form(default=500, ge=100, le=2000),
    chunk_overlap: int = Form(default=50, ge=0, le=500),
    doc_type: str = Form(default="auto", description="文档类型: auto(自动检测) / contract / regulation / faq / resume / general"),
):
    """上传文档文件，自动解析、切分、嵌入并建立检索索引。

    支持 5 种切分策略：
    - contract:   按条款编号切（第X条）
    - regulation: 按法规条/款切（双层）
    - faq:        按问答对切
    - resume:     按简历模块切（教育/工作/项目）
    - general:    通用递归切分
    - auto:       自动检测文档类型
    """
    pipeline = get_pipeline()

    resolved_doc_type = None if doc_type == "auto" else doc_type
    saved_count = 0
    imported_files = 0

    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in _LOADERS:
            continue

        imported_files += 1

        # 保存上传文件（消毒文件名，防止路径穿越）
        safe_name = Path(file.filename).name  # 去掉任何目录前缀
        docs_dir = Path(settings.docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        temp_path = docs_dir / safe_name
        content = await file.read()
        temp_path.write_bytes(content)

        # 用现有 loader 读取
        loader = _LOADERS.get(suffix)
        if loader:
            try:
                docs = loader(temp_path)
            except Exception as e:
                logger.warning("加载文件 %s 失败: %s", file.filename, e)
                continue

            # 切分（传入文档类型）
            chunks = chunk_documents(
                docs,
                doc_type=resolved_doc_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            dict_docs = [
                {"text": chunk.page_content, "metadata": chunk.metadata}
                for chunk in chunks
            ]
            pipeline.add_documents(dict_docs)
            saved_count += len(dict_docs)

    # 重建索引（自动持久化）
    pipeline.build_index()

    return {
        "imported_files": imported_files,
        "chunks_created": saved_count,
        "total_documents": pipeline.retriever.document_count,
    }


@router.post("/search", response_model=RagSearchResponse, summary="向量 + BM25 双路搜索")
async def search(req: RagSearchRequest):
    """执行向量 + BM25 双路检索（含 RRF 融合），返回原始结果（不经过 LLM）。"""
    pipeline = get_pipeline()
    start = time.time()
    results = pipeline.retriever.search(req.query, top_k=req.top_k)
    latency = (time.time() - start) * 1000

    return RagSearchResponse(
        results=[
            RagSearchResult(text=text, metadata=meta, score=score)
            for text, meta, score in results
        ],
        latency_ms=round(latency, 2),
    )


@router.post("/query", response_model=RagQueryResponse, summary="完整 RAG 查询")
async def rag_query(req: RagQueryRequest):
    """执行完整 RAG 流程：检索 → 重排序 → LLM 生成。"""
    pipeline = get_pipeline()
    result = await pipeline.query(
        question=req.question,
        top_k=req.top_k,
        rerank_top_k=req.rerank_top_k,
        system_prompt=req.system_prompt,
    )
    return RagQueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )


@router.post("/query/stream", summary="流式 SSE RAG 查询")
async def rag_query_stream(req: RagQueryRequest):
    """SSE (Server-Sent Events) 流式 RAG 查询。"""
    pipeline = get_pipeline()

    # 检索 + 重排序
    retrieved = pipeline.retriever.search(req.question, top_k=req.top_k)
    sources = []
    if retrieved:
        docs_for_rerank = [
            {"text": text, "metadata": meta, "retrieval_score": score}
            for text, meta, score in retrieved
        ]
        reranked = pipeline.reranker.rerank_with_docs(
            req.question, docs_for_rerank, top_k=req.rerank_top_k
        )
        context = "\n\n".join(
            f"[来源 {i+1}] {d['text']}" for i, d in enumerate(reranked)
        )
        sources = [
            {"content": d["text"][:500], "score": d.get("rerank_score", 0)}
            for d in reranked
        ]
    else:
        context = ""

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        prompt = (
            f"基于以下参考信息，回答用户的问题。\n\n"
            f"参考信息：\n{context}\n\n"
            f"用户问题：{req.question}\n\n"
            f"请基于参考信息给出准确、详细的回答。"
        )
        system_prompt = req.system_prompt or "你是一个专业的知识库问答助手，请基于提供的信息准确回答。"

        try:
            async for token in pipeline.generator.stream_generate(
                prompt=prompt, system_prompt=system_prompt
            ):
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/index/status", response_model=IndexStatusResponse, summary="索引状态")
async def index_status():
    """查看当前索引的统计信息。"""
    pipeline = get_pipeline()
    return IndexStatusResponse(
        document_count=pipeline.retriever.document_count,
        index_type=settings.faiss_index_type,
        dimension=pipeline.embedding_model.dim,
    )


@router.post("/index/rebuild", summary="强制重建索引")
async def rebuild_index():
    """清空当前索引并基于 docs_dir 中的所有文件重建。"""
    from app.data.loader import load_documents

    pipeline = get_pipeline()
    pipeline.retriever.clear()

    docs = load_documents()
    chunks = chunk_documents(docs)
    dict_docs = [
        {"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    ]
    pipeline.add_documents(dict_docs)
    pipeline.build_index()

    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "total_documents": pipeline.retriever.document_count,
    }


@router.post("/agent/query", summary="多智能体 RAG 查询（含改写+规划+护栏）")
async def agent_query(req: RagQueryRequest):
    """执行完整的多 Agent RAG 查询流程：
    查询改写 → 检索规划 → 多步检索 → 答案合成 → 护栏校验。

    相比 /api/rag/query，增加了查询改写、多子查询并行检索、幻觉检测和安全检查。
    """
    from app.agent.rag_agent import rag_agent_workflow

    result = await rag_agent_workflow.run(req.question)
    return result
