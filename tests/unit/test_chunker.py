"""chunker 文档切分单元测试"""

from __future__ import annotations

from langchain_core.documents import Document

from app.data.chunker import chunk_documents


class TestChunker:
    def test_chunk_splits_chinese_text(self):
        """应正确切分中文文本。"""
        text = "第一段内容。第二段内容。第三段内容。" * 20  # ~300 chars
        docs = [Document(page_content=text, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """每个 chunk 不应显著超过 chunk_size。"""
        text = "这是一个测试文档，" * 100
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= 300  # 允许一些容差

    def test_empty_document_returns_empty(self):
        """空文档应返回空列表。"""
        docs = [Document(page_content="")]
        chunks = chunk_documents(docs)
        # 空内容可能返回一个空 chunk 或空列表
        non_empty = [c for c in chunks if c.page_content.strip()]
        assert len(non_empty) == 0

    def test_empty_list_returns_empty(self):
        """空文档列表应返回空列表。"""
        chunks = chunk_documents([])
        assert chunks == []

    def test_metadata_preserved(self):
        """切分后每个 chunk 应保留原始 metadata。"""
        docs = [Document(page_content="测试内容 " * 30, metadata={"source": "test.pdf", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"
            assert chunk.metadata.get("page") == 1

    def test_multiple_documents(self):
        """多个文档应都能被切分。"""
        docs = [
            Document(page_content="文档A内容 " * 20, metadata={"id": "a"}),
            Document(page_content="文档B内容 " * 20, metadata={"id": "b"}),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        ids = {c.metadata.get("id") for c in chunks}
        assert "a" in ids
        assert "b" in ids

    def test_chinese_boundary_splits(self):
        """中文标点符号应作为分隔边界。"""
        text = "第一部分。第二部分！第三部分？第四部分；第五部分。"
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=2)
        # 应该产生多个 chunk（在标点处分割）
        assert len(chunks) >= 2
