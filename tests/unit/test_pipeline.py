"""RAGPipeline 单元测试"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_query_retrieve_rerank_generate(self, sample_docs, mock_embedding_model, mock_llm_generator, mock_reranker_model):
        """完整 RAG 查询应返回 answer / sources / latency_ms。"""
        from app.rag.pipeline import RAGPipeline
        from app.rag.retriever import DualRetriever

        retriever = DualRetriever(embedding_model=mock_embedding_model)
        retriever.add_documents(sample_docs)
        retriever.build_index()

        pipeline = RAGPipeline(
            embedding_model=mock_embedding_model,
            retriever=retriever,
            reranker=mock_reranker_model,
            generator=mock_llm_generator,
        )

        result = await pipeline.query("什么是 FAISS？", top_k=3, rerank_top_k=2)
        assert "answer" in result
        assert "sources" in result
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)

    @pytest.mark.asyncio
    async def test_query_no_docs_falls_back_to_llm(self, mock_embedding_model, mock_llm_generator):
        """无文档时 query 应降级为直接使用 LLM 回答（不报错）。"""
        from app.rag.pipeline import RAGPipeline
        from app.rag.retriever import DualRetriever

        retriever = DualRetriever(embedding_model=mock_embedding_model)
        pipeline = RAGPipeline(
            embedding_model=mock_embedding_model,
            retriever=retriever,
            generator=mock_llm_generator,
        )

        result = await pipeline.query("测试问题", top_k=3)
        assert "answer" in result
        assert result["sources"] == []  # 无文档，无来源

    @pytest.mark.asyncio
    async def test_add_and_index(self, sample_docs, mock_embedding_model):
        """add_documents + build_index 应使文档可检索。"""
        from app.rag.pipeline import RAGPipeline
        from app.rag.retriever import DualRetriever

        retriever = DualRetriever(embedding_model=mock_embedding_model)
        pipeline = RAGPipeline(
            embedding_model=mock_embedding_model,
            retriever=retriever,
        )
        pipeline.add_documents(sample_docs)
        pipeline.build_index()
        assert pipeline.retriever.document_count == len(sample_docs)

    @pytest.mark.asyncio
    async def test_match_resume_jd_structured(self, mock_embedding_model, mock_llm_generator):
        """简历-JD 匹配应返回 MatchResult。"""
        from app.rag.pipeline import RAGPipeline
        from app.rag.retriever import DualRetriever

        retriever = DualRetriever(embedding_model=mock_embedding_model)
        pipeline = RAGPipeline(
            embedding_model=mock_embedding_model,
            retriever=retriever,
            generator=mock_llm_generator,
        )

        with patch.object(mock_llm_generator, 'generate_structured') as mock_gs:
            mock_gs.return_value = {
                "match_score": 0.75,
                "matched_skills": ["Python", "FAISS"],
                "missing_skills": ["Docker"],
                "analysis": "匹配度较好",
                "suggestions": ["学习Docker"],
            }
            result = await pipeline.match_resume_jd_structured(
                resume_text="Python 和 FAISS 经验丰富的开发者",
                jd_text="需要 Python、FAISS、Docker 技能",
            )
            assert result.match_score == 0.75
            assert "Python" in result.matched_skills
            assert "Docker" in result.missing_skills
