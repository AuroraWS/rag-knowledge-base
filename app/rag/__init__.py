"""
RAG 核心模块
- embeddings: Embedding 模型封装
- retriever: 双路检索（BM25 + 向量）+ RRF 融合
- reranker: bge-reranker 重排序
- generator: DeepSeek LLM 调用
- pipeline: RAG 完整流程编排
"""
