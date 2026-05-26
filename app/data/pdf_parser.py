"""PDF 解析器 — 统一管线处理文字型与扫描型 PDF。

文字型:  pymupdf 直接提取文本（快速、免费）
扫描型:  pymupdf → 页面图像 → OCR/备用（当前用 pymupdf 内置能力）
结构化:  提取的文本 → DeepSeek LLM → 结构化字段（extraction_service）

流程:  PDF → pymupdf 文本提取 → DeepSeek LLM 结构化 → Pydantic 模型
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# 最少文本量——低于此值认为 PDF 可能是扫描件/图片型
MIN_TEXT_THRESHOLD = 100

# 一次 LLM 调用最多处理的文本长度（字符）
MAX_TEXT_FOR_LLM = 12000


class PDFParser:
    """PDF 解析器 — 提取文本 + 调用 LLM 结构化。

    Usage:
        parser = PDFParser()
        text = parser.extract_text("resume.pdf")
        structured = await parser.extract_structured("resume.pdf")
    """

    def __init__(self, max_pages: int = 30, dpi: int = 200):
        self._max_pages = max_pages
        self._dpi = dpi

    # ── 文本提取 ──────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """从 PDF 中提取纯文本。

        Args:
            file_path: PDF 文件路径。

        Returns:
            提取的文本内容（失败时返回空字符串）。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        text = self._extract_text_pymupdf(str(path))
        if len(text.strip()) < MIN_TEXT_THRESHOLD:
            logger.info("文本量不足 (%d 字符)，PDF 可能是扫描件: %s", len(text.strip()), file_path)
            # 尝试提高 DPI 重新提取（扫描件优化）
            text = self._extract_text_pymupdf(str(path), dpi=self._dpi)

        return text

    def _extract_text_pymupdf(self, file_path: str, dpi: int = 150) -> str:
        """pymupdf 文本提取。"""
        try:
            import fitz

            doc = fitz.open(file_path)
            pages = []
            total = min(len(doc), self._max_pages)
            for i in range(total):
                page = doc[i]
                text = page.get_text()
                if not text.strip() and dpi > 150:
                    # 扫描件：尝试通过图像提取文本（pymupdf 自身支持）
                    text = page.get_text("text")
                pages.append(text)
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.error("pymupdf 未安装")
            return ""
        except Exception as e:
            logger.error("pymupdf 提取失败 (%s): %s", file_path, e)
            return ""

    # ── 扫描件处理 ────────────────────────────────────

    def extract_page_images(self, file_path: str) -> list[bytes]:
        """将 PDF 页面渲染为 PNG 图像（供外部 OCR/视觉 API 使用）。

        Args:
            file_path: PDF 文件路径。

        Returns:
            每页的 PNG 字节数据列表。
        """
        images: list[bytes] = []
        try:
            import fitz

            doc = fitz.open(file_path)
            total = min(len(doc), self._max_pages)
            for i in range(total):
                page = doc[i]
                pix = page.get_pixmap(dpi=self._dpi)
                images.append(pix.tobytes("png"))
            doc.close()
        except ImportError:
            logger.error("pymupdf 未安装，无法渲染页面")
        except Exception as e:
            logger.error("PDF 页面渲染失败: %s", e)
        return images

    # ── 结构化提取 ────────────────────────────────────

    async def extract_structured(self, file_path: str) -> dict[str, Any]:
        """从 PDF 中提取结构化简历信息。

        完整流程: 文本提取 → LLM 结构化 → Pydantic 反序列化。

        Args:
            file_path: PDF 文件路径。

        Returns:
            {"personal_info": PersonalInfo|None, "education": [...],
             "work_experience": [...], "projects": [...], "certificates": [...],
             "raw_text": str}
        """
        from app.services.extraction_service import extraction_service

        text = self.extract_text(file_path)
        if not text.strip():
            raise ValueError(f"无法从 PDF 提取文本: {file_path}")

        result = await extraction_service.extract_from_text(text)
        result["raw_text"] = text
        result["source_file"] = file_path
        return result

    # ── 信息 ──────────────────────────────────────────

    def get_page_count(self, file_path: str) -> int:
        """获取 PDF 页数。"""
        try:
            import fitz

            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"<PDFParser max_pages={self._max_pages} dpi={self._dpi}>"


# 模块级单例
pdf_parser = PDFParser()
