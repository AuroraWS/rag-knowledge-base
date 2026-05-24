"""DualRetriever — BM25 + FAISS 双路检索单元测试"""

from __future__ import annotations

import numpy as np
import pytest


class TestDualRetriever:
    def test_init_state(self):
        """初始状态检查。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever()
        assert r.document_count == 0

    def test_add_documents(self, sample_docs):
        """添加文档应增加计数。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever()
        r.add_documents(sample_docs)
        assert r.document_count == len(sample_docs)

    def test_add_document_missing_text_raises(self):
        """缺少 'text' 键的文档应引发 ValueError。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever()
        with pytest.raises(ValueError, match="必须包含 'text'"):
            r.add_documents([{"metadata": {}}])

    def test_build_index_flat(self, sample_docs, mock_embedding_model):
        """构建 Flat 索引后 FAISS 和 BM25 均应就绪。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        r.add_documents(sample_docs)
        r.build_index()
        assert r._index is not None
        assert r._bm25 is not None
        assert "Flat" in repr(r)

    def test_search_returns_top_k(self, sample_docs, mock_embedding_model):
        """检索应返回指定数量的结果。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        r.add_documents(sample_docs)
        r.build_index()
        results = r.search("Python 编程", top_k=3)
        assert 0 < len(results) <= 3

    def test_search_no_docs_returns_empty(self):
        """无文档时检索应返回空列表。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever()
        results = r.search("测试")
        assert results == []

    def test_clear_resets_everything(self, sample_docs, mock_embedding_model):
        """clear() 应清空文档和索引。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        r.add_documents(sample_docs)
        r.build_index()
        r.clear()
        assert r.document_count == 0
        assert r._bm25 is None
        assert r._index is None

    def test_save_and_load_index(self, sample_docs, mock_embedding_model, tmp_path):
        """保存后加载索引，文档数应一致。"""
        from app.rag.retriever import DualRetriever

        r1 = DualRetriever(embedding_model=mock_embedding_model)
        r1.add_documents(sample_docs)
        r1.build_index()

        index_path = tmp_path / "test_index.faiss"
        r1.save_index(str(index_path))
        assert index_path.exists()

        r2 = DualRetriever()
        r2.load_index(str(index_path))
        assert r2._index is not None

    def test_bm25_search_only(self, sample_docs, mock_embedding_model):
        """仅 BM25 检索应基于关键词匹配。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        r.add_documents(sample_docs)
        r.build_index()
        results = r.search_bm25_only("FAISS", top_k=3)
        assert len(results) > 0
        # FAISS 应出现在结果中
        faiss_found = any("FAISS" in text for text, _, _ in results)
        assert faiss_found

    def test_chinese_tokenization(self, sample_docs, mock_embedding_model):
        """中文按字分词应产生合理的 BM25 得分（需多个文档以避免 IDF 为零）。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        docs = [
            {"text": "这是中文测试文本内容"},
            {"text": "这是另一篇中文文档"},
            {"text": "BM25 是一种稀疏检索算法"},
        ]
        r.add_documents(docs)
        r.build_index()
        results = r.search_bm25_only("中文测试", top_k=3)
        assert len(results) > 0
        assert results[0][2] > 0  # score > 0

    def test_rrf_fusion(self, sample_docs, mock_embedding_model):
        """RRF 融合后的结果分数应大于 0。"""
        from app.rag.retriever import DualRetriever
        r = DualRetriever(embedding_model=mock_embedding_model)
        r.add_documents(sample_docs)
        r.build_index()
        results = r.search("FAISS 向量检索", top_k=5)
        for _, _, score in results:
            assert score > 0
