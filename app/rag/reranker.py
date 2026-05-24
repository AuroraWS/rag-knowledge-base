"""
重排序模块 — 基于 sentence-transformers CrossEncoder (BAAI/bge-reranker-v2-m3)

对检索结果进行精细化的交叉编码重排序，提高最终召回的精准度。
lazy loading: 模型在首次调用 rerank 时加载。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """CrossEncoder 重排序器（BGE Reranker 系列，lazy loading）"""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.reranker_model
        self._model = None  # lazy load

    # ── 属性 ──────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── 模型加载 ─────────────────────────────────────────

    def _get_model(self):
        """lazy load CrossEncoder 模型"""
        if self._model is not None:
            return self._model

        import os
        from sentence_transformers import CrossEncoder

        if settings.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

        logger.info("加载 reranker 模型: %s (local_only=%s)", self._model_name, settings.local_files_only)
        self._model = CrossEncoder(
            self._model_name,
            local_files_only=settings.local_files_only,
        )
        logger.info("reranker 模型加载完成")
        return self._model

    # ── 重排序 ───────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        对候选文本列表进行重排序，返回 (text, score) 按分数降序排列。

        Parameters
        ----------
        query : str
            查询文本
        candidates : list[str]
            候选文本列表
        top_k : int
            返回 top_k 个结果

        Returns
        -------
        list[tuple[str, float]]
            重排序后的 (text, score) 列表，按分数降序
        """
        if not candidates:
            return []

        model = self._get_model()

        # CrossEncoder 需要 (query, candidate) 对
        pairs = [[query, cand] for cand in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        # 确保 scores 是列表
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = list(scores)

        # 按分数降序排列
        scored_pairs = list(zip(candidates, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]

    def rerank_with_docs(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        对候选文档列表（dict），保留 metadata 进行重排序。

        Parameters
        ----------
        query : str
            查询文本
        docs : list[dict]
            文档列表，每个 dict 含 'text' 和可选的 'metadata'/'score'
        top_k : int
            返回 top_k 个结果

        Returns
        -------
        list[dict]
            重排序后的文档列表，每个 dict 包含 'text', 'metadata', 'score'
        """
        if not docs:
            return []

        candidates = [d["text"] for d in docs]
        reranked = self.rerank(query, candidates, top_k=len(candidates))

        # 建一个 text -> original_doc 的映射以保留 metadata
        doc_map = {d["text"]: d for d in docs}

        results: list[dict[str, Any]] = []
        for text, score in reranked[:top_k]:
            original = doc_map.get(text, {})
            results.append({
                "text": text,
                "metadata": original.get("metadata", {}),
                "rerank_score": float(score),
            })

        return results

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "lazy"
        return f"<Reranker {self._model_name} ({loaded})>"
