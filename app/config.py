"""
RAG 企业知识库智能问答系统 — 配置模块
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件加载"""

    # LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # Embedding & Reranker
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000

    # 检索
    top_k: int = 5
    rerank_top_k: int = 3

    # 路径
    knowledge_base_dir: str = "knowledge_base"
    faiss_index_dir: str = "faiss_index"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
