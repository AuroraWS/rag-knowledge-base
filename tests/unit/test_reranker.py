"""Reranker 单元测试"""

from __future__ import annotations

import pytest


class TestReranker:
    def test_rerank_returns_top_k(self, mock_reranker_model):
        """重排序应返回正确数量的结果。"""
        candidates = [
            {"text": "文档 A — 相关内容", "metadata": {"id": "a"}, "retrieval_score": 0.8},
            {"text": "文档 B — 相关", "metadata": {"id": "b"}, "retrieval_score": 0.7},
            {"text": "文档 C — 不太相关", "metadata": {"id": "c"}, "retrieval_score": 0.6},
        ]
        result = mock_reranker_model.rerank_with_docs("查询", candidates, top_k=2)
        assert 0 < len(result) <= 2

    def test_rerank_scores_descending(self, mock_reranker_model):
        """重排序分数应降序排列。"""
        candidates = [
            {"text": f"文档 {i}", "metadata": {}, "retrieval_score": 0.5}
            for i in range(5)
        ]
        result = mock_reranker_model.rerank_with_docs("查询", candidates, top_k=5)
        scores = [doc.get("rerank_score", 0) for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_preserves_metadata(self, mock_reranker_model):
        """重排序后应保留原始 metadata。"""
        candidates = [
            {"text": "测试文本", "metadata": {"source": "test.pdf", "page": 3}, "retrieval_score": 0.9},
        ]
        result = mock_reranker_model.rerank_with_docs("查询", candidates, top_k=1)
        assert result[0]["metadata"]["source"] == "test.pdf"
        assert result[0]["metadata"]["page"] == 3

    def test_empty_candidates(self, mock_reranker_model):
        """空候选列表应返回空结果。"""
        result = mock_reranker_model.rerank_with_docs("查询", [], top_k=3)
        assert result == []
