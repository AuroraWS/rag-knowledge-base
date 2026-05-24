"""
双路检索模块 — BM25 稀疏检索 + FAISS 稠密检索 + RRF 融合

流程:
1. BM25 (rank_bm25) 对文档做关键词匹配
2. FAISS (IndexFlatIP) 对文档做向量相似度检索
3. RRF (Reciprocal Rank Fusion, k=60) 融合两路分数
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import settings
from app.rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# RRF 常数
RRF_K = 60


class DualRetriever:
    """双路检索器: BM25 + FAISS + RRF 融合"""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self._embedding_model = embedding_model or EmbeddingModel()
        self._bm25 = None  # rank_bm25.BM25Okapi
        self._index = None  # faiss index (Flat / HNSW / IVF / HNSW_PQ)
        self._index_type: str = "Flat"  # 实际使用的索引类型
        self._documents: list[dict[str, Any]] = []  # 原始文档 [{text, metadata}]
        self._dim: Optional[int] = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def embedding_model(self) -> EmbeddingModel:
        return self._embedding_model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self._embedding_model.dim
        return self._dim

    @property
    def document_count(self) -> int:
        return len(self._documents)

    # ── 文档管理 ─────────────────────────────────────────

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """添加文档。每个 dict 必须有 'text' 键，可选 'metadata'"""
        for doc in docs:
            if "text" not in doc:
                raise ValueError("每个文档必须包含 'text' 字段")
            self._documents.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            })
        logger.info("已添加 %d 篇文档，总计 %d 篇", len(docs), len(self._documents))

    def clear(self) -> None:
        """清空所有文档和索引"""
        self._documents.clear()
        self._bm25 = None
        self._index = None

    # ── 索引构建 ─────────────────────────────────────────

    def build_index(self) -> None:
        """基于当前所有文档构建 BM25 + FAISS 索引"""
        if not self._documents:
            logger.warning("没有文档可建索引")
            return

        texts = [d["text"] for d in self._documents]

        # ── BM25 ───────────────────────────────────────
        self._build_bm25(texts)

        # ── FAISS ──────────────────────────────────────
        logger.info("构建 FAISS 索引，维度: %d, 文档数: %d, 类型: %s", self.dim, len(texts), settings.faiss_index_type)
        embeddings = self._embedding_model.encode_documents(texts)
        self._index = self._build_faiss_index(embeddings)
        logger.info("FAISS 索引构建完成 (类型: %s)", self._index_type)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """根据配置构建 FAISS 索引，支持 Flat / HNSW / IVF / HNSW_PQ。

        训练类索引（IVF, HNSW_PQ）需要足够样本才能 train()，
        样本不足或训练失败时自动回退到 IndexFlatIP。
        """
        import faiss

        index_type = settings.faiss_index_type
        dim = self.dim
        n = embeddings.shape[0]

        try:
            if index_type == "HNSW":
                index = faiss.IndexHNSWFlat(dim, settings.hnsw_m)
                index.hnsw.efConstruction = settings.hnsw_ef_construction
                index.hnsw.efSearch = settings.hnsw_ef_search
                self._index_type = "HNSW"

            elif index_type == "IVF":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, settings.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
                if n >= settings.ivf_nlist:
                    index.train(embeddings)
                else:
                    logger.warning("文档数 (%d) 少于 IVF 聚类数 (%d)，回退到 Flat", n, settings.ivf_nlist)
                    index = faiss.IndexFlatIP(dim)
                    self._index_type = "Flat"
                    index.add(embeddings)
                    return index
                index.nprobe = settings.ivf_nprobe
                self._index_type = "IVF"

            elif index_type == "HNSW_PQ":
                m = min(settings.hnsw_m, dim)  # HNSW M 不能超过 dim
                index = faiss.IndexHNSWPQ(dim, m, dim // 4, 8)
                index.hnsw.efConstruction = settings.hnsw_ef_construction
                index.hnsw.efSearch = settings.hnsw_ef_search
                if n >= 256:
                    index.train(embeddings)
                else:
                    logger.warning("文档数 (%d) 不足以训练 HNSW_PQ (需 >= 256)，回退到 Flat", n)
                    index = faiss.IndexFlatIP(dim)
                    self._index_type = "Flat"
                    index.add(embeddings)
                    return index
                self._index_type = "HNSW_PQ"

            else:
                # "Flat" 或其他未知值，默认暴力搜索
                index = faiss.IndexFlatIP(dim)
                self._index_type = "Flat"

            index.add(embeddings)
            return index

        except (RuntimeError, ValueError) as e:
            logger.warning("FAISS %s 索引创建失败: %s，回退到 Flat", index_type, e)
            index = faiss.IndexFlatIP(dim)
            self._index_type = "Flat"
            index.add(embeddings)
            return index

    def _build_bm25(self, texts: list[str]) -> None:
        """构建 BM25 索引（中文按字切分）"""
        from rank_bm25 import BM25Okapi

        logger.info("构建 BM25 索引，文档数: %d", len(texts))
        tokenized = [list(text) for text in texts]  # 按字切分
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引构建完成")

    # ── 检索 ───────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """
        检索 query，返回 (text, metadata, score) 列表，按融合分数降序。

        Parameters
        ----------
        query : str
            查询文本
        top_k : int
            最终返回的结果数
        bm25_weight : float
            BM25 分数在融合中的权重
        vector_weight : float
            向量分数在融合中的权重

        Returns
        -------
        list[tuple[str, dict, float]]
            (text, metadata, fused_score)
        """
        if not self._documents:
            return []

        n_docs = len(self._documents)

        # ── BM25 检索 ──────────────────────────────────
        bm25_scores: Optional[list[float]] = None
        if self._bm25 is not None:
            tokenized_query = list(query)
            bm25_raw = self._bm25.get_scores(tokenized_query)
            # 归一化 BM25 分数到 [0, 1]
            b_max = bm25_raw.max()
            if b_max > 0:
                bm25_scores = (bm25_raw / b_max).tolist()
            else:
                bm25_scores = [0.0] * n_docs

        # ── 向量检索 ───────────────────────────────────
        vector_scores: Optional[list[float]] = None
        if self._index is not None:
            q_vec = self._embedding_model.encode_query(query)
            vec_sim, vec_idx = self._index.search(q_vec, min(top_k * 3, n_docs))
            vec_sim = vec_sim[0].tolist()
            vec_idx = vec_idx[0].tolist()

            # 构建全量向量分数列表（未召回的设为 0）
            v = [0.0] * n_docs
            for idx, score in zip(vec_idx, vec_sim):
                if idx >= 0 and idx < n_docs:
                    v[idx] = score
            vector_scores = v

        # ── RRF 融合 ───────────────────────────────────
        fused_scores = np.zeros(n_docs, dtype=np.float32)

        if bm25_scores is not None:
            # RRF ranking from BM25
            bm25_ranks = np.argsort(np.argsort(-np.array(bm25_scores)))
            for i in range(n_docs):
                fused_scores[i] += bm25_weight * (1.0 / (RRF_K + bm25_ranks[i]))

        if vector_scores is not None:
            vec_ranks = np.argsort(np.argsort(-np.array(vector_scores)))
            for i in range(n_docs):
                fused_scores[i] += vector_weight * (1.0 / (RRF_K + vec_ranks[i]))

        # ── 排序取 top_k ───────────────────────────────
        top_indices = np.argsort(-fused_scores)[:top_k]

        results: list[tuple[str, dict[str, Any], float]] = []
        for idx in top_indices:
            doc = self._documents[idx]
            score = float(fused_scores[idx])
            if score > 0:
                results.append((doc["text"], doc["metadata"], score))

        return results

    def search_bm25_only(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, dict[str, Any], float]]:
        """仅用 BM25 检索"""
        if self._bm25 is None:
            return []
        tokenized_query = list(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-np.array(scores))[:top_k]
        return [
            (self._documents[i]["text"], self._documents[i]["metadata"], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def search_vector_only(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, dict[str, Any], float]]:
        """仅用向量检索"""
        if self._index is None:
            return []
        q_vec = self._embedding_model.encode_query(query)
        vec_sim, vec_idx = self._index.search(q_vec, min(top_k, len(self._documents)))
        return [
            (
                self._documents[int(idx)]["text"],
                self._documents[int(idx)]["metadata"],
                float(sim),
            )
            for idx, sim in zip(vec_idx[0], vec_sim[0])
            if idx >= 0
        ]

    def save_index(self, path: str) -> None:
        """保存 FAISS 索引到磁盘"""
        import faiss
        if self._index is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, path)
            logger.info("FAISS 索引已保存到 %s", path)

    def load_index(self, path: str) -> None:
        """从磁盘加载 FAISS 索引"""
        import faiss
        p = Path(path)
        if p.exists():
            self._index = faiss.read_index(str(p))
            self._dim = self._index.d
            logger.info("FAISS 索引已从 %s 加载（维度: %d）", path, self._dim)

    def __repr__(self) -> str:
        bm25_status = "built" if self._bm25 is not None else "none"
        faiss_status = f"{self._index_type}" if self._index is not None else "none"
        return (
            f"<DualRetriever docs={len(self._documents)} "
            f"BM25={bm25_status} FAISS={faiss_status}>"
        )
