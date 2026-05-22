"""
RAG 核心模块 — 检索增强生成流水线

组件:
- EmbeddingModel: 文本嵌入（bge-small-zh-v1.5）
- DualRetriever:  双路检索（BM25 + FAISS + RRF 融合）
- Reranker:        CrossEncoder 重排序（bge-reranker-v2-m3）
- LLMGenerator:    DeepSeek API 调用（OpenAI 兼容）
- RAGPipeline:     完整 RAG 流程编排 + 简历-JD 匹配
"""

from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import DualRetriever
from app.rag.reranker import Reranker
from app.rag.generator import LLMGenerator
from app.rag.pipeline import RAGPipeline

__all__ = [
    "EmbeddingModel",
    "DualRetriever",
    "Reranker",
    "LLMGenerator",
    "RAGPipeline",
]
