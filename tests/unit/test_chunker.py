"""chunker 文档切分单元测试"""

from __future__ import annotations

from langchain_core.documents import Document

from app.data.chunker import chunk_documents, detect_doc_type


class TestChunker:
    # ── general 通用切分 ────────────────────────────

    def test_chunk_splits_chinese_text(self):
        """应正确切分中文文本。"""
        text = "第一段内容。第二段内容。第三段内容。" * 20
        docs = [Document(page_content=text, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """每个 chunk 不应显著超过 chunk_size。"""
        text = "这是一个测试文档，" * 100
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= 300

    def test_empty_document_returns_empty(self):
        """空文档应返回空列表。"""
        docs = [Document(page_content="")]
        chunks = chunk_documents(docs)
        non_empty = [c for c in chunks if c.page_content.strip()]
        assert len(non_empty) == 0

    def test_empty_list_returns_empty(self):
        """空文档列表应返回空列表。"""
        chunks = chunk_documents([])
        assert chunks == []

    def test_metadata_preserved(self):
        """切分后每个 chunk 应保留原始 metadata + 新增 doc_type。"""
        docs = [Document(page_content="测试内容 " * 30, metadata={"source": "test.pdf", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"
            assert "doc_type" in chunk.metadata
            assert "split_method" in chunk.metadata

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
        text = "第一部分。第二部分！第三部分？第四部分；第五部分。" * 4
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, chunk_size=30, chunk_overlap=2)
        assert len(chunks) >= 2

    # ── 文档类型检测 ─────────────────────────────────

    def test_detect_contract(self):
        """包含甲方乙方+条款编号 → 合同。"""
        text = "甲方：王爽\n第一条 项目概况\n第二条 权利义务\n乙方签章："
        assert detect_doc_type(text) == "contract"

    def test_detect_regulation(self):
        """包含第X条+法规关键词 → 法规。"""
        text = "第一章 总则\n第一条 根据《XX条例》制定本办法。\n第二条 本办法适用于..."
        assert detect_doc_type(text) == "regulation"

    def test_detect_faq(self):
        """包含多个 Q: 标记 → FAQ。"""
        text = "Q: 什么是RAG？\nA: RAG是检索增强生成。\nQ: 如何部署？\nA: 使用Docker。"
        assert detect_doc_type(text) == "faq"

    def test_detect_resume(self):
        """包含教育+工作经历+技能 → 简历。"""
        text = "教育经历\n2019-2023 XX大学\n工作经历\n2023-至今 YY公司\n技能\nPython, Java"
        assert detect_doc_type(text) == "resume"

    def test_detect_general(self):
        """无特殊标记 → 通用。"""
        text = "这是一段普通的描述性文本，没有任何特殊的文档结构标记。"
        assert detect_doc_type(text) == "general"

    # ── 合同切分 ─────────────────────────────────────

    def test_contract_chunking(self):
        """合同应按条款编号切分。"""
        text = "第一条 项目范围\n本项目包含以下内容。\n第二条 付款方式\n甲方应在签署后支付。"
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, doc_type="contract")
        assert len(chunks) >= 2
        assert all(c.metadata["doc_type"] == "contract" for c in chunks)

    # ── FAQ 切分 ─────────────────────────────────────

    def test_faq_chunking(self):
        """FAQ 应按问答对切分。"""
        text = "Q: 第一个问题？\nA: 第一个答案。\n\nQ: 第二个问题？\nA: 第二个答案。"
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, doc_type="faq")
        assert len(chunks) >= 2
        assert all(c.metadata["doc_type"] == "faq" for c in chunks)

    # ── 简历切分 ─────────────────────────────────────

    def test_resume_chunking(self):
        """简历应按模块切分。"""
        text = (
            "教育经历\n2019-2023 XX大学 计算机科学\n"
            "工作经历\n2023-至今 YY公司 软件工程师\n"
            "项目经历\nRAG知识库系统\n技能\nPython, FAISS, Docker"
        )
        docs = [Document(page_content=text)]
        chunks = chunk_documents(docs, doc_type="resume")
        assert len(chunks) >= 2
        assert all(c.metadata["doc_type"] == "resume" for c in chunks)
        sections = {c.metadata.get("section", "") for c in chunks}
        assert "教育经历" in sections or any("教育" in s for s in sections)
