"""EmbeddingModel 单元测试"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingModel:
    def test_initialization_does_not_load_model(self):
        """初始化 EmbeddingModel 不应立即加载 SentenceTransformer。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            from app.rag.embeddings import EmbeddingModel
            model = EmbeddingModel()
            mock_st.assert_not_called()

    def test_dim_triggers_loading(self):
        """访问 dim 属性应触发模型加载。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 512
            mock_st.return_value = mock_instance

            from app.rag.embeddings import EmbeddingModel
            model = EmbeddingModel()
            dim = model.dim
            assert dim == 512
            mock_st.assert_called_once()

    def test_encode_adds_query_prefix(self):
        """encode_query 应添加 BGE 查询前缀。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)
            mock_st.return_value = mock_instance

            from app.rag.embeddings import EmbeddingModel
            model = EmbeddingModel()
            model.encode_query("测试查询")
            call_args = mock_instance.encode.call_args[0][0]
            assert "为这个句子生成表示以用于检索相关文章" in str(call_args)

    def test_encode_documents_no_prefix(self):
        """encode_documents 不应添加查询前缀，且应做 L2 归一化。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(3, 384).astype(np.float32)
            mock_st.return_value = mock_instance

            from app.rag.embeddings import EmbeddingModel
            model = EmbeddingModel()
            model.encode_documents(["文档1", "文档2", "文档3"])
            # 验证 encode 被调用时传了 normalize_embeddings=True
            _, kwargs = mock_instance.encode.call_args
            assert kwargs.get("normalize_embeddings") is True

    def test_similarity_shape(self, mock_embedding_model):
        """similarity 应返回正确形状的相似度矩阵。"""
        q = np.random.rand(2, 384).astype(np.float32)
        d = np.random.rand(5, 384).astype(np.float32)
        sim = mock_embedding_model.similarity(q, d)
        assert sim.shape == (2, 5)

    def test_singleton_behavior(self):
        """同一个实例应只加载一次模型。"""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_instance

            from app.rag.embeddings import EmbeddingModel
            model = EmbeddingModel()
            _ = model.dim
            _ = model.dim  # 第二次访问
            assert mock_st.call_count == 1  # 只加载一次
