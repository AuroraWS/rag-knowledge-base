"""共享测试夹具 — mock embedding / reranker / llm / sample_docs"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── 测试数据路径 ───────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_json(name: str):
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 样本文档 ──────────────────────────────────────


@pytest.fixture
def sample_docs():
    """中文文档集合，用于检索器测试。"""
    return [
        {"text": "Python 是一种高级编程语言，广泛用于数据科学、机器学习和 Web 开发。", "metadata": {"source": "doc1", "title": "Python概述"}},
        {"text": "FAISS 是 Facebook AI 开发的向量检索库，支持大规模相似性搜索和聚类。", "metadata": {"source": "doc2", "title": "FAISS介绍"}},
        {"text": "RAG 是检索增强生成技术，结合了信息检索和文本生成的能力。", "metadata": {"source": "doc3", "title": "RAG概述"}},
        {"text": "LangChain 是一个用于构建 LLM 应用的框架，提供了链式调用和 Agent 功能。", "metadata": {"source": "doc4", "title": "LangChain框架"}},
        {"text": "DeepSeek 是一个国产大语言模型，支持中英文对话和代码生成。", "metadata": {"source": "doc5", "title": "DeepSeek模型"}},
        {"text": "深度学习是机器学习的一个重要分支，使用多层神经网络进行特征提取。PyTorch 是最流行的深度学习框架之一。", "metadata": {"source": "doc6"}},
        {"text": "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率计算文档相关性。", "metadata": {"source": "doc7"}},
        {"text": "HNSW（Hierarchical Navigable Small World）是一种高效的近似最近邻搜索算法。FAISS 中实现了 HNSW 索引。", "metadata": {"source": "doc8"}},
        {"text": "向量检索通过计算查询向量与文档向量之间的相似度来查找相关内容。余弦相似度和内积是常用的度量方式。", "metadata": {"source": "doc9"}},
        {"text": "大语言模型在自然语言处理任务中表现出色，包括文本生成、翻译、摘要和问答。", "metadata": {"source": "doc10"}},
    ]


# ── Mock Embedding Model ──────────────────────────


@pytest.fixture
def mock_embedding_model():
    """返回 EmbeddingModel 的 mock 实例，不加载真实模型。"""
    from unittest.mock import MagicMock, patch

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(10, 384).astype(np.float32)
        mock_st.return_value = mock_instance

        from app.rag.embeddings import EmbeddingModel
        model = EmbeddingModel()
        yield model


# ── Mock LLM Generator ─────────────────────────────


@pytest.fixture
def mock_llm_generator():
    """返回 LLMGenerator 的 mock 实例，不调用真实 API。"""
    from unittest.mock import AsyncMock, MagicMock, patch

    with patch("app.rag.generator.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "这是一个测试回答，基于提供的参考信息生成。"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client.return_value = mock_instance

        from app.rag.generator import LLMGenerator
        gen = LLMGenerator(api_key="test-key")
        yield gen


# ── Mock Reranker ──────────────────────────────────


@pytest.fixture
def mock_reranker_model():
    """返回 Reranker 的 mock 实例，不加载真实 CrossEncoder。"""
    from unittest.mock import MagicMock, patch

    with patch("sentence_transformers.CrossEncoder") as mock_ce:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_ce.return_value = mock_instance

        from app.rag.reranker import Reranker
        reranker = Reranker()
        yield reranker


# ── Golden Q/A pairs ──────────────────────────────


@pytest.fixture
def golden_qa():
    """加载黄金问答对，用于回归测试。"""
    return _load_json("golden_qa.json")


# ── pytest 标记 ────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 标记慢速测试（需要真实模型或 API 调用）")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="运行慢速测试")
    parser.addoption("--runintegration", action="store_true", help="运行集成测试（需要真实模型）")


def pytest_collection_modifyitems(config, items):
    # 默认跳过 slow 和 integration 测试
    if not config.getoption("--runslow"):
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(pytest.mark.skip(reason="需要 --runslow 以运行慢速测试"))
    if not config.getoption("--runintegration"):
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(pytest.mark.skip(reason="需要 --runintegration 以运行集成测试"))
