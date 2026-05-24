"""
文档切分器 - 使用 RecursiveCharacterTextSplitter + 中文友好分隔符。

chunk_size / chunk_overlap 默认从 config.py 读取，
调用方可传入参数覆盖。
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """将文档切分为更小的块，使用中文友好分隔符。"""
    cs = chunk_size if chunk_size is not None else settings.chunk_size
    co = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)
