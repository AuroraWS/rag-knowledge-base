"""文档切分器 — 根据文档类型选择最优切分策略。

支持的文档类型与策略：
- contract (合同):    按条款编号切（第X条、第X章），保证每块是一个完整条款
- regulation (法规):  双层策略——按"条"切分，附带上层"章/节"上下文
- faq (问答):         按 Q&A 对切分，每块包含完整的一问一答
- resume (简历):      按简历模块切分（教育/工作/项目/技能），保留板块标题
- general (通用):     回退到 RecursiveCharacterTextSplitter，中文友好分隔符

Usage:
    from app.data.chunker import chunk_documents
    chunks = chunk_documents(docs, doc_type="contract")
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# ── 合同：按条款编号切分 ─────────────────────────────

# 匹配 "第X条" "第X章" "第X节" 等中文条款标记
_CLAUSE_PATTERN = re.compile(
    r"(?=第\s*[一二三四五六七八九十百千\d]+\s*(?:条|章|节|款|项|部分|编))"
)

# 匹配英文合同条款: "Article 1" "Section 2.1" "Clause 3"
_CLAUSE_EN_PATTERN = re.compile(
    r"(?=(?:Article|Section|Clause|PART|CHAPTER)\s+\d+)",
    re.IGNORECASE,
)


def _chunk_by_clause(documents: List[Document]) -> List[Document]:
    """按条款编号切分——适用于合同、协议类文档。"""
    chunks: List[Document] = []
    for doc in documents:
        text = doc.page_content
        # 先尝试中文条款
        parts = _CLAUSE_PATTERN.split(text)
        if len(parts) <= 1:
            # 再尝试英文条款
            parts = _CLAUSE_EN_PATTERN.split(text)

        if len(parts) <= 1:
            # 没有找到条款标记，按段落切
            parts = text.split("\n\n")

        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) < 10:
                continue  # 跳过太短的片段（如页码、空白）
            chunks.append(Document(
                page_content=part,
                metadata={
                    **doc.metadata,
                    "doc_type": "contract",
                    "chunk_index": i,
                    "split_method": "clause",
                },
            ))
    return chunks


# ── 法规：双层结构（章+条） ──────────────────────────

# 匹配法规的 "第X章" 作为一级标记
_CHAPTER_PATTERN = re.compile(r"(第\s*[一二三四五六七八九十百千\d]+\s*章\s*.*?)\n")
# 匹配法规的 "第X条" 作为二级标记
_ARTICLE_PATTERN = re.compile(r"(?=第\s*[一二三四五六七八九十百千\d]+\s*条)")


def _chunk_by_article(documents: List[Document]) -> List[Document]:
    """按法规的「章→条」双层结构切分，每条附加上层章标题。

    策略：
    1. 先按"章"切粗块
    2. 每章内按"条"切细块
    3. 细块的 content 包含所属章标题 + 原文
    """
    chunks: List[Document] = []
    for doc in documents:
        text = doc.page_content
        # 按"章"切分
        chapter_parts = _CHAPTER_PATTERN.split(text)

        if len(chapter_parts) <= 1:
            # 没有"章"，直接按"条"切
            articles = _ARTICLE_PATTERN.split(text)
            for i, article in enumerate(articles):
                article = article.strip()
                if len(article) < 10:
                    continue
                chunks.append(Document(
                    page_content=article,
                    metadata={
                        **doc.metadata,
                        "doc_type": "regulation",
                        "chunk_index": i,
                        "split_method": "article",
                    },
                ))
            continue

        # 有章结构：第一段是章标题前的文字（如有），跳过
        offset = 0
        if chapter_parts and not _CHAPTER_PATTERN.match(
            chapter_parts[0] if chapter_parts else ""
        ):
            offset = 1

        chunk_idx = 0
        for ci in range(offset, len(chapter_parts) - 1, 2):
            chapter_title = chapter_parts[ci].strip() if ci < len(chapter_parts) else ""
            chapter_body = chapter_parts[ci + 1] if ci + 1 < len(chapter_parts) else ""

            # 章内按"条"切
            articles = _ARTICLE_PATTERN.split(chapter_body)
            for article in articles:
                article = article.strip()
                if len(article) < 10:
                    continue
                # 附上所属章标题作为上下文
                content = f"【{chapter_title}】{article}" if chapter_title else article
                chunks.append(Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "doc_type": "regulation",
                        "chapter": chapter_title,
                        "chunk_index": chunk_idx,
                        "split_method": "article",
                    },
                ))
                chunk_idx += 1

    return chunks


# ── FAQ：按 Q&A 对切分 ──────────────────────────────

_QA_SPLIT_PATTERN = re.compile(
    r"(?=(?:Q\s*[：:.\d]|问\s*[：:]|【问】|问题\s*\d*\s*[：:]|FAQ\s*\d*\s*[：:]))",
    re.IGNORECASE,
)

# 匹配 A:/答: 把 QA 对内部也标记出来（但不切）
_QA_ANSWER_PATTERN = re.compile(
    r"(?:A\s*[：:.]|答\s*[：:]|【答】|回答\s*[：:])",
    re.IGNORECASE,
)


def _chunk_by_qa_pair(documents: List[Document]) -> List[Document]:
    """按 Q&A 对切分——保证每个 chunk 包含完整的一问一答。"""
    chunks: List[Document] = []
    for doc in documents:
        text = doc.page_content
        pairs = _QA_SPLIT_PATTERN.split(text)

        if len(pairs) <= 1:
            # 没有 Q&A 结构，回退到按双换行切（FAQ 里一个 QA 对通常是一段）
            pairs = text.split("\n\n")

        for i, pair in enumerate(pairs):
            pair = pair.strip()
            if len(pair) < 15:
                continue
            # 判断是否同时包含问题和答案
            has_answer = bool(_QA_ANSWER_PATTERN.search(pair))
            chunks.append(Document(
                page_content=pair,
                metadata={
                    **doc.metadata,
                    "doc_type": "faq",
                    "chunk_index": i,
                    "is_complete_qa": has_answer,
                    "split_method": "qa_pair",
                },
            ))
    return chunks


# ── 简历：按模块切分 ────────────────────────────────

_RESUME_SECTION_PATTERN = re.compile(
    r"(?=(?:教育(?:经历|背景)?|工作(?:经历|经验)?|实习(?:经历)?|"
    r"项目(?:经历|经验)?|技能(?:特长)?|证书(?:资质)?|"
    r"个人(?:信息|简介|概况)?|联系方式|"
    r"自我(?:评价|介绍|描述)))"
)


def _chunk_by_section(documents: List[Document]) -> List[Document]:
    """按简历模块标题切分，每个模块独立成块，保留模块标题作为上下文。"""
    chunks: List[Document] = []
    for doc in documents:
        text = doc.page_content
        sections = _RESUME_SECTION_PATTERN.split(text)

        if len(sections) <= 1:
            # 没有明确的模块标题，用 RecursiveCharacterTextSplitter 作为 fallback
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=30,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                length_function=len,
            )
            sub = splitter.split_documents([doc])
            for s in sub:
                s.metadata["doc_type"] = "resume"
                s.metadata["split_method"] = "recursive_fallback"
            chunks.extend(sub)
            continue

        # 跳过空的第一段，从 index=1 开始每段都是一个 section
        # re.split 以零宽断言切分，结果中每个非空元素以某个 section 标题开头
        valid_sections = [s.strip() for s in sections if s.strip()]
        for i, section_text in enumerate(valid_sections):
            if len(section_text) < 15:
                continue

            # 提取 section 标题（第一行的模块名）
            first_line = section_text.split("\n")[0].strip()
            header = first_line if len(first_line) <= 15 else "其他模块"

            # 如果单个 section 太长，做二次切分
            if len(section_text) > 600:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=40,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                    length_function=len,
                )
                temp_doc = Document(page_content=section_text, metadata=doc.metadata)
                sub_chunks = splitter.split_documents([temp_doc])
                for sc in sub_chunks:
                    sc.metadata.update({
                        "doc_type": "resume",
                        "section": header,
                        "split_method": "section_then_recursive",
                    })
                    chunks.append(sc)
            else:
                chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "doc_type": "resume",
                        "section": header,
                        "split_method": "section",
                    },
                ))

    return chunks


# ── 通用：RecursiveCharacterTextSplitter ──────────────

def _chunk_recursive(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """通用切分器——中文友好分隔符，适用于无固定结构的文档。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    for c in chunks:
        c.metadata["doc_type"] = "general"
        c.metadata["split_method"] = "recursive"
    return chunks


# ── 自动检测文档类型 ──────────────────────────────────

def detect_doc_type(text: str) -> str:
    """根据文本内容自动推断文档类型。

    检测优先级：合同/法规 > FAQ > 简历 > general

    Args:
        text: 文档全文内容（前 3000 字符即可）。

    Returns:
        doc_type: contract / regulation / faq / resume / general
    """
    preview = text[:3000]

    # 合同特征：包含"甲方""乙方""合同""协议" + 条款编号
    contract_signals = ["甲方", "乙方", "合同", "协议", "签署", "签章"]
    contract_score = sum(1 for s in contract_signals if s in preview)
    has_clause_nums = bool(_CLAUSE_PATTERN.search(preview)) or bool(_CLAUSE_EN_PATTERN.search(preview))
    if contract_score >= 2 and has_clause_nums:
        return "contract"

    # 法规特征：包含"第X条" + "章" 结构
    has_articles = bool(_ARTICLE_PATTERN.search(preview))
    has_chapters = bool(_CHAPTER_PATTERN.search(preview))
    regulation_signals = ["法规", "条例", "办法", "规定", "通知", "公告"]
    reg_score = sum(1 for s in regulation_signals if s in preview)
    if has_articles and (has_chapters or reg_score >= 1):
        return "regulation"

    # FAQ 特征：Q&A 格式
    qa_count = len(_QA_SPLIT_PATTERN.findall(preview))
    if qa_count >= 2:
        return "faq"

    # 简历特征：包含多个简历模块标题
    resume_signals = ["教育", "工作经历", "项目经历", "技能", "自我评价"]
    resume_score = sum(1 for s in resume_signals if s in preview)
    if resume_score >= 2:
        return "resume"

    return "general"


# ── 主入口 ──────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    doc_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """根据文档类型选择最优切分策略。

    Args:
        documents: LangChain Document 列表（来自 loader）。
        doc_type: 文档类型（contract/regulation/faq/resume/general）。
                  为 None 时自动检测（基于第一篇文档的内容）。
        chunk_size: 通用切分器的块大小（仅对 general/fallback 有效）。
        chunk_overlap: 通用切分器的重叠量。

    Returns:
        切分后的 Document 列表，每个 chunk 的 metadata 含 doc_type 和 split_method。
    """
    if not documents:
        return []

    cs = chunk_size if chunk_size is not None else settings.chunk_size
    co = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    # 自动检测文档类型
    if doc_type is None:
        # 拼接所有文档的前面部分用于检测
        combined_text = " ".join(d.page_content[:2000] for d in documents[:3])
        doc_type = detect_doc_type(combined_text)
        logger.info("自动检测文档类型: %s", doc_type)

    logger.info("切分策略: %s (chunk_size=%d, overlap=%d)", doc_type, cs, co)

    # 策略分发
    if doc_type == "contract":
        chunks = _chunk_by_clause(documents)
    elif doc_type == "regulation":
        chunks = _chunk_by_article(documents)
    elif doc_type == "faq":
        chunks = _chunk_by_qa_pair(documents)
    elif doc_type == "resume":
        chunks = _chunk_by_section(documents)
    else:
        chunks = _chunk_recursive(documents, chunk_size=cs, chunk_overlap=co)

    # 过滤空块和过短块
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 10]

    logger.info("切分完成: %d 篇文档 → %d 个 chunks", len(documents), len(chunks))
    return chunks
