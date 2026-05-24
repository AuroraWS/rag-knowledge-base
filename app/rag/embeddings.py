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

        import os
        from pathlib import Path
        from sentence_transformers import SentenceTransformer

        # 如果配置了 HF 镜像，设置环境变量
        if settings.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

        logger.info("加载 embedding 模型: %s (local_only=%s)", self._model_name, settings.local_files_only)
        try:
            self._model = SentenceTransformer(
                self._model_name,
                local_files_only=settings.local_files_only,
            )
        except TypeError as e:
            # sentence_transformers 5.x 与旧模型格式不兼容时，尝试修补 Pooling 配置
            if "Pooling" not in str(e):
                raise
            logger.warning("模型加载失败（Pooling 配置缺失），尝试自动修补: %s", e)
            self._patch_pooling_config()
            self._model = SentenceTransformer(
                self._model_name,
                local_files_only=True,
            )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("embedding 模型加载完成, 维度: %d", self._dim)
        return self._model

    def _patch_pooling_config(self):
        """为旧格式模型创建缺失的 1_Pooling/config.json。

        sentence_transformers >= 5.x 要求 Pooling 模块配置文件包含
        word_embedding_dimension 字段，但旧版模型（v2.x）没有该文件。
        """
        import json
        from huggingface_hub import try_to_load_from_cache

        # 从缓存中查找模型快照路径
        from sentence_transformers.util.file_io import load_file_path
        modules_path = load_file_path(
            self._model_name, "modules.json", local_files_only=True
        )
        if modules_path is None:
            logger.warning("无法从缓存中找到 modules.json，跳过 Pooling 配置修补")
            return
        snapshot_dir = Path(modules_path).parent
        pooling_dir = snapshot_dir / "1_Pooling"
        pooling_config = pooling_dir / "config.json"

        if not pooling_config.exists():
            # 从 transformer config 获取 hidden_size
            from transformers import AutoConfig
            transformer_config = AutoConfig.from_pretrained(
                str(snapshot_dir), local_files_only=True
            )
            hidden_size = getattr(transformer_config, "hidden_size", 512)

            pooling_dir.mkdir(parents=True, exist_ok=True)
            pooling_config.write_text(
                json.dumps({
                    "word_embedding_dimension": hidden_size,
                    "pooling_mode_cls_token": False,
                    "pooling_mode_mean_tokens": True,
                    "pooling_mode_max_tokens": False,
                    "pooling_mode_mean_sqrt_len_tokens": False,
                    "pooling_mode_weightedmean_tokens": False,
                    "pooling_mode_lasttoken": False,
                    "include_prompt": True,
                }),
                encoding="utf-8",
            )
            logger.info("已创建 Pooling 配置文件: %s (dim=%d)", pooling_config, hidden_size)

    # ── 编码 ─────────────────────────────────────────────

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码文本，返回归一化后的嵌入矩阵 (n, dim)"""
        model = self._get_model()
        if not texts:
            assert self._dim is not None
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
