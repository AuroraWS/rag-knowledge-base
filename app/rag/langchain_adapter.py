"""
LangChain LCEL RAG 检索链路适配器
===================================
封装完整 RAG 管道：Embedding → FAISS 检索 → format → LLM 生成。

支持的调用方式：
- invoke(query) → str          同步
- stream(query) → Generator    流式
- ainvoke(query) → str         异步
- batch(queries) → list[str]   批量
"""

import os
from typing import Generator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from app.config import settings

# 系统提示词
SYSTEM_PROMPT = (
    "你是一个专业的企业知识库助手，回答基于提供的参考上下文。"
    "如果上下文中没有足够信息，请如实告知，不要编造。"
    "回答要求：准确、简洁、条理清晰，使用中文。"
)

HUMAN_TEMPLATE = "参考上下文：\n\n{context}\n\n问题：{question}"


def _format_docs(docs: List[Document]) -> str:
    """将检索到的文档拼接为上下文字符串。"""
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "未知来源")
        parts.append(f"[参考{i} | {src}]\n{doc.page_content}")
    return "\n\n".join(parts)


class _RetrieverCache:
    """FAISS vector store + retriever 懒加载缓存。"""

    def __init__(self):
        self._embeddings = None
        self._retriever = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def retriever(self):
        if self._retriever is None:
            index_path = os.path.join(settings.faiss_index_dir, "index.faiss")
            if os.path.exists(index_path):
                vectorstore = FAISS.load_local(
                    settings.faiss_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                from app.data.loader import load_documents
                from app.data.chunker import chunk_documents

                docs = load_documents()
                chunks = chunk_documents(docs)
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
                os.makedirs(settings.faiss_index_dir, exist_ok=True)
                vectorstore.save_local(settings.faiss_index_dir)

            self._retriever = vectorstore.as_retriever(
                search_kwargs={"k": settings.top_k}
            )
        return self._retriever


_cache = _RetrieverCache()


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        model="deepseek-chat",
        temperature=0.3,
        streaming=True,
    )


def _build_chain():
    """构建 LCEL RAG 管道。"""
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    chain = (
        {
            "context": _cache.retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


class LangChainRAGAdapter:
    """LangChain LCEL RAG 适配器。

    Usage:
        adapter = LangChainRAGAdapter()
        answer = adapter.invoke("什么是 RAG？")
        for chunk in adapter.stream("什么是 RAG？"):
            print(chunk, end="")
        answers = adapter.batch(["问题1", "问题2"])
        answer = await adapter.ainvoke("什么是 RAG？")
    """

    def __init__(self):
        self.chain = _build_chain()

    # ── (d) 多种调用方式 ──────────────────────────────

    def invoke(self, query: str) -> str:
        """同步调用，返回完整回答。"""
        return self.chain.invoke(query)

    def stream(self, query: str) -> Generator[str, None, None]:
        """流式输出，逐步 yield token 字符串。"""
        for chunk in self.chain.stream(query):
            yield chunk

    async def ainvoke(self, query: str) -> str:
        """异步调用。"""
        return await self.chain.ainvoke(query)

    def batch(self, queries: List[str]) -> List[str]:
        """批量调用，返回与输入顺序对应的回答列表。"""
        return self.chain.batch(queries)
