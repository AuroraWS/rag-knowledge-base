"""
智能招聘助手 — 配置模块

所有配置从环境变量加载，支持 .env 文件。
使用 pydantic_settings.BaseSettings 自动管理类型转换和默认值。
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件加载"""

    # ── LLM ──────────────────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── Embedding & Reranker ─────────────────────────
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    local_files_only: bool = False             # True=仅用本地缓存，不联网下载
    hf_endpoint: str = ""                      # HF 镜像（如 https://hf-mirror.com），留空用官方

    # ── 服务 ─────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── 检索 ─────────────────────────────────────────
    top_k: int = 5
    rerank_top_k: int = 3

    # ── FAISS 索引 ─────────────────────────────────────
    faiss_index_type: str = "Flat"          # Flat / HNSW / IVF / HNSW_PQ
    hnsw_m: int = 32                        # HNSW: M (每个节点的连接数)
    hnsw_ef_construction: int = 200         # HNSW: 构建时的搜索宽度
    hnsw_ef_search: int = 64                # HNSW: 搜索时的搜索宽度
    ivf_nlist: int = 256                    # IVF: 聚类中心数
    ivf_nprobe: int = 10                    # IVF: 搜索时探测的聚类数

    # ── 文档切分 ─────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── 数据目录 ─────────────────────────────────────
    knowledge_base_dir: str = "data/docs"
    faiss_index_dir: str = "data/faiss_index"
    sqlite_path: str = "data/applications/applications.db"
    profile_dir: str = "data/profile"
    docs_dir: str = "data/docs"
    memory_dir: str = "data/memory"
    log_dir: str = "data/logs"

    # ── WeChat bot ───────────────────────────────────
    wechat_appid: str = ""
    wechat_appsecret: str = ""

    # ── 定时任务 ─────────────────────────────────────
    daily_log_enabled: bool = True
    daily_log_time: str = "22:00"
    schedule_review_time: str = "09:30"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()

# 尽早设置 HF 镜像，避免被 huggingface_hub 缓存旧值
if settings.hf_endpoint:
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
