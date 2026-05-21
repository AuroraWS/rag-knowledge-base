"""
文本嵌入模块 — 基于 sentence-transformers (BAAI/bge-small-zh-v1.5)

lazy loading: 模型在首次调用 encode 时加载，避免导入时阻塞。
输出向量已归一化（L2 norm），适用于余弦相似度（内积即相似度）。
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# bge 系列模型需要在 query 前加特定前缀以获得最佳效果
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingModel:
    """sentence-transformers 嵌入模型封装（BGE 系列，lazy loading）"""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.embedding_model
        self._model = None  # lazy load
        self._dim: Optional[int] = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        if value != self._model_name:
            self._model = None
            self._dim = None
            self._model_name = value

    @property
    def dim(self) -> int:
        """嵌入向量维度（触发加载）"""
        if self._dim is None:
            _ = self._get_model()
        assert self._dim is not None
        return self._dim

    # ── 模型加载 ─────────────────────────────────────────

    def _get_model(self):
        """lazy load sentence-transformers 模型"""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        cache_folder = str(Path(settings.embedding_model).parent)
        logger.info("加载 embedding 模型: %s (cache: %s)", self._model_name, cache_folder)
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=cache_folder,
            local_files_only=True,  # 优先使用本地缓存
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("embedding 模型加载完成, 维度: %d", self._dim)
        return self._model

    # ── 编码 ─────────────────────────────────────────────

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码文本，返回归一化后的嵌入矩阵 (n, dim)"""
        model = self._get_model()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """编码单个 query（自动添加 bge 查询前缀），返回形状 (1, dim)"""
        prefixed = f"{BGE_QUERY_PREFIX}{text}"
        return self.encode([prefixed])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """编码文档/语料（不加前缀），返回 (n, dim)"""
        return self.encode(texts)

    # ── 工具 ─────────────────────────────────────────────

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算两个归一化向量之间的余弦相似度"""
        return a @ b.T

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "lazy"
        return f"<EmbeddingModel {self._model_name} ({loaded})>"
