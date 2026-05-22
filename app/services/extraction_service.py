"""文档提取服务 — 从 PDF / 图片中解析结构化简历信息。

流程：
1. 先用规则/OCR 提取文本
2. 再用 LLM 进行结构化验证和补全
3. 返回 Pydantic 模型：PersonalInfo, Education, WorkExperience, Project, Certificate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.models.schema import (
    Certificate,
    Education,
    PersonalInfo,
    Project,
    WorkExperience,
)
from app.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

# ── 系统提示词 ──────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """你是一位专业的简历解析专家。请从提供的文档文本中提取结构化信息。

输出必须是一个 JSON 对象（不要包含 markdown 代码块标记），包含以下字段：

{
    "personal_info": {
        "name": "姓名",
        "phone": "手机号",
        "email": "邮箱",
        "wechat": "微信号或null",
        "target_location": ["期望工作地点"],
        "target_salary": "期望薪资如'15k-20k'或null",
        "earliest_start_date": "最快到岗时间或null"
    },
    "education": [
        {
            "school": "学校名称",
            "degree": "学士/硕士/博士",
            "major": "专业",
            "start_date": "入学时间如'2023.09'",
            "end_date": "毕业时间如'2025.01'",
            "gpa": 3.8 或 null,
            "is_overseas": true/false
        }
    ],
    "work_experience": [
        {
            "company": "公司名称",
            "department": "部门或null",
            "title": "职位",
            "start_date": "开始时间",
            "end_date": "结束时间或null",
            "is_current": false,
            "responsibilities": ["职责1", "职责2"],
            "achievements": ["成果1", "成果2"],
            "tech_stack": ["技术1", "技术2"]
        }
    ],
    "projects": [
        {
            "name": "项目名称",
            "role": "角色",
            "start_date": "开始时间或null",
            "end_date": "结束时间或null",
            "description": "项目描述",
            "tech_stack": ["技术1"],
            "highlights": ["亮点1", "亮点2"]
        }
    ],
    "certificates": [
        {
            "name": "证书名称",
            "issuer": "发证机构",
            "date": "获得日期或null"
        }
    ]
}

注意：
- 未找到的字段用 null 或空列表
- 日期尽量保持原始格式
- 如果文档内容不是简历，返回空结构
"""


class ExtractionService:
    """文档提取服务 — 将 PDF/图片解析为结构化简历数据。"""

    def __init__(self, generator: Optional[LLMGenerator] = None) -> None:
        self._generator = generator or LLMGenerator()

    # ── PDF 提取 ─────────────────────────────────────

    async def extract_from_pdf(self, file_path: str) -> dict[str, Any]:
        """从 PDF 文件中提取结构化简历信息。

        Args:
            file_path: PDF 文件路径。

        Returns:
            包含 personal_info, education, work_experience, projects, certificates 的字典。
            每个字段为 Pydantic 模型或模型列表。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件: {file_path}")

        # Step 1: 使用 PyMuPDF 提取文本
        text = self._extract_text_pymupdf(str(path))

        if not text.strip():
            logger.warning("PyMuPDF 未提取到文本，尝试备用提取: %s", file_path)
            text = self._extract_text_fallback(str(path))

        if not text.strip():
            raise ValueError(f"无法从 PDF 提取文本: {file_path}")

        # Step 2: LLM 结构化解析
        return await self._parse_with_llm(text)

    def _extract_text_pymupdf(self, file_path: str) -> str:
        """使用 PyMuPDF (fitz) 提取 PDF 文本。"""
        try:
            import fitz  # type: ignore[import-untyped]

            doc = fitz.open(file_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append(text)
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyMuPDF 未安装，跳过 PDF 文本提取")
            return ""
        except Exception as e:
            logger.error("PyMuPDF 提取失败: %s", e)
            return ""

    def _extract_text_fallback(self, file_path: str) -> str:
        """备用文本提取（纯文本文件或 python-docx）。"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    # ── 图片提取（占位） ─────────────────────────────

    async def extract_from_image(self, file_path: str) -> dict[str, Any]:
        """从图片中提取结构化信息（当前为占位实现）。

        Args:
            file_path: 图片文件路径。

        Returns:
            与 extract_from_pdf 相同的结构。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

        # TODO: 集成 OCR (如 PaddleOCR / Tesseract) 进行图片文字识别
        logger.warning("图片提取功能尚未实现: %s", file_path)
        return {
            "personal_info": None,
            "education": [],
            "work_experience": [],
            "projects": [],
            "certificates": [],
        }

    # ── LLM 解析 ─────────────────────────────────────

    async def _parse_with_llm(self, text: str) -> dict[str, Any]:
        """使用 LLM 从文本中提取结构化数据。"""
        prompt = f"""请从以下文档文本中提取简历信息：

===== 文档内容 =====
{text[:8000]}
===== 文档结束 =====

请按 JSON 格式输出结构化的简历信息。"""

        try:
            result = await self._generator.generate_structured(
                prompt=prompt,
                system_prompt=EXTRACT_SYSTEM_PROMPT,
                temperature=0.1,
            )

            # 确保结果是 dict
            if isinstance(result, str):
                result = json.loads(result)

            # 反序列化为 Pydantic 模型
            return self._deserialize_result(result)

        except Exception as e:
            logger.error("LLM 解析失败: %s", e, exc_info=True)
            # 降级返回空结构
            return {
                "personal_info": None,
                "education": [],
                "work_experience": [],
                "projects": [],
                "certificates": [],
            }

    def _deserialize_result(self, raw: dict) -> dict[str, Any]:
        """将 LLM 返回的 dict 反序列化为 Pydantic 模型。"""
        result: dict[str, Any] = {}

        # personal_info
        pi_raw = raw.get("personal_info")
        if pi_raw and isinstance(pi_raw, dict) and pi_raw.get("name"):
            result["personal_info"] = PersonalInfo(**pi_raw)
        else:
            result["personal_info"] = None

        # education
        edu_list = []
        for e in raw.get("education", []):
            if isinstance(e, dict) and e.get("school"):
                try:
                    edu_list.append(Education(**e))
                except Exception as ex:
                    logger.warning("教育经历解析跳过: %s", ex)
        result["education"] = edu_list

        # work_experience
        work_list = []
        for w in raw.get("work_experience", []):
            if isinstance(w, dict) and w.get("company"):
                try:
                    work_list.append(WorkExperience(**w))
                except Exception as ex:
                    logger.warning("工作经历解析跳过: %s", ex)
        result["work_experience"] = work_list

        # projects
        proj_list = []
        for p in raw.get("projects", []):
            if isinstance(p, dict) and p.get("name"):
                try:
                    proj_list.append(Project(**p))
                except Exception as ex:
                    logger.warning("项目经历解析跳过: %s", ex)
        result["projects"] = proj_list

        # certificates
        cert_list = []
        for c in raw.get("certificates", []):
            if isinstance(c, dict) and c.get("name"):
                try:
                    cert_list.append(Certificate(**c))
                except Exception as ex:
                    logger.warning("证书解析跳过: %s", ex)
        result["certificates"] = cert_list

        return result

    async def close(self) -> None:
        """释放资源。"""
        await self._generator.close()


# 模块级单例
extraction_service = ExtractionService()
