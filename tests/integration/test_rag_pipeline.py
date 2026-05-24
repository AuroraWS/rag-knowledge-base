"""RAG Pipeline 集成测试 — 使用真实 embedding 模型"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.integration
class TestRAGPipelineIntegration:
    @pytest.mark.asyncio
    async def test_real_embedding_mock_llm(self, sample_docs):
        """使用真实 embedding + mock LLM 的集成测试。"""
        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.add_documents(sample_docs)
        pipeline.build_index()

        # Mock LLM 调用
        with patch.object(pipeline.generator, 'generate') as mock_gen:
            mock_gen.return_value = "这是一个集成测试的回答"
            result = await pipeline.query("什么是 FAISS？", top_k=3, rerank_top_k=2)
            assert "answer" in result
            assert len(result["sources"]) > 0  # 应有检索结果
            assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_pipeline_latency_under_threshold(self, sample_docs):
        """RAG 检索+重排序延迟应在合理范围内（不含 LLM）。"""
        import time
        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.add_documents(sample_docs)
        pipeline.build_index()

        # 只测检索部分延迟
        start = time.time()
        results = pipeline.retriever.search("FAISS", top_k=5)
        elapsed = (time.time() - start) * 1000

        assert len(results) > 0
        assert elapsed < 5000, f"检索延迟 {elapsed:.0f}ms 超过阈值 5000ms"

    @pytest.mark.asyncio
    async def test_bm25_vs_vector_consistency(self, sample_docs):
        """BM25 和向量检索都应返回非空结果。"""
        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.add_documents(sample_docs)
        pipeline.build_index()

        bm25_results = pipeline.retriever.search_bm25_only("FAISS 向量检索", top_k=3)
        vec_results = pipeline.retriever.search_vector_only("FAISS 向量检索", top_k=3)

        assert len(bm25_results) > 0, "BM25 检索不应为空"
        assert len(vec_results) > 0, "向量检索不应为空"
