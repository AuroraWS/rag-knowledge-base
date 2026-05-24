"""智能招聘助手 — 简历信息管理 API

路由前缀: /api/profile

提供简历五个模块的完整 CRUD + 文档提取确认流程：
- 文档上传 → AI 提取 → 预览 → 确认/修正 → 入库
- 个人基本信息 (PersonalInfo)
- 教育经历 (Education)
- 工作经历 (WorkExperience)
- 项目经历 (Project)
- 证书/资质 (Certificate)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.models.schema import (
    Certificate,
    Education,
    PersonalInfo,
    Project,
    WorkExperience,
)
from app.storage.profile_store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profile", tags=["简历信息管理"])


# ── 依赖注入 ─────────────────────────────────────


def get_profile_store() -> ProfileStore:
    from app.storage.profile_store import profile_store
    return profile_store


# ── 请求/响应模型 ─────────────────────────────────


class ExtractionPreview(BaseModel):
    """文档提取预览——包含 AI 提取结果 + 原始文本供用户核对。"""
    personal_info: Optional[dict[str, Any]] = Field(None, description="提取的个人信息")
    education: list[dict[str, Any]] = Field(default_factory=list)
    work_experience: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    certificates: list[dict[str, Any]] = Field(default_factory=list)
    raw_text_preview: str = Field("", description="原文前500字供对照")
    page_count: int = Field(0, description="PDF 页数")


class ConfirmSaveRequest(BaseModel):
    """用户确认/修正后的数据入库请求。"""
    personal_info: Optional[dict[str, Any]] = Field(None, description="确认/修正后的个人信息")
    education: list[dict[str, Any]] = Field(default_factory=list)
    work_experience: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    certificates: list[dict[str, Any]] = Field(default_factory=list)
    modified_fields: list[str] = Field(default_factory=list, description="用户手动修改的字段名列表")
    source_file: str = Field("", description="来源文件名")


# ── 上传 + AI 提取（预览，不保存） ────────────────────


@router.post(
    "/upload/extract",
    summary="上传简历 → AI 提取预览（不保存）",
    response_model=ExtractionPreview,
)
async def upload_and_extract(
    file: UploadFile = File(..., description="简历文件（支持 PDF/DOCX）"),
):
    """上传简历文件，AI 自动提取结构化信息并返回预览。

    此端点**不会**保存数据。用户查看预览后：
    - 满意 → 调用 POST /upload/confirm 确认入库
    - 有误 → 修正后调用 POST /upload/confirm 提交修正版本
    """
    # Step 1: 保存临时文件
    suffix = os.path.splitext(file.filename or "resume.pdf")[1].lower()
    if suffix not in (".pdf", ".docx", ".txt", ".md"):
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}，请上传 PDF / DOCX / TXT / MD")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 2: 提取文本 + LLM 结构化
        if suffix == ".pdf":
            from app.data.pdf_parser import pdf_parser
            text = pdf_parser.extract_text(tmp_path)
            page_count = pdf_parser.get_page_count(tmp_path)
        elif suffix == ".docx":
            from app.data.loader import _load_docx
            from pathlib import Path
            docs = _load_docx(Path(tmp_path))
            text = "\n\n".join(d.page_content for d in docs) if docs else ""
            page_count = 0
        else:
            # TXT / MD 直接读取
            text = content.decode("utf-8", errors="ignore")
            page_count = 0

        if not text.strip():
            raise HTTPException(status_code=422, detail="无法从文件中提取文本内容")

        from app.services.extraction_service import extraction_service
        extracted = await extraction_service.extract_from_text(text)

        # Step 3: 构建预览（Pydantic 模型 → dict）
        preview = ExtractionPreview(
            personal_info=extracted.get("personal_info").model_dump() if extracted.get("personal_info") else None,
            education=[e.model_dump() for e in (extracted.get("education") or [])],
            work_experience=[w.model_dump() for w in (extracted.get("work_experience") or [])],
            projects=[p.model_dump() for p in (extracted.get("projects") or [])],
            certificates=[c.model_dump() for c in (extracted.get("certificates") or [])],
            raw_text_preview=text[:500],
            page_count=page_count,
        )
        return preview

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("提取失败: %s", e)
        raise HTTPException(status_code=500, detail=f"AI 提取失败: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── 确认入库 ──────────────────────────────────────


@router.post("/upload/confirm", summary="确认/修正提取结果并入库")
async def confirm_and_save(req: ConfirmSaveRequest):
    """用户确认（或修正后）的提取结果入库。

    传入确认/修正后的数据 + modified_fields 列表，
    系统会标记哪些字段来自 AI、哪些被手动修改。
    """
    store = get_profile_store()

    saved: dict[str, Any] = {}

    # 保存个人信息
    if req.personal_info and req.personal_info.get("name"):
        try:
            info = PersonalInfo(**req.personal_info)
            store.update_personal_info(info)
            saved["personal_info"] = info.model_dump()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"个人信息格式错误: {e}")

    # 保存教育经历（追加模式）
    for edu_dict in req.education:
        try:
            edu = Education(**edu_dict)
            store.add_education(edu)
            saved.setdefault("education", []).append(edu.model_dump())
        except Exception as e:
            logger.warning("教育经历保存跳过: %s", e)

    # 保存工作经历（追加模式）
    for work_dict in req.work_experience:
        try:
            work = WorkExperience(**work_dict)
            store.add_work_experience(work)
            saved.setdefault("work_experience", []).append(work.model_dump())
        except Exception as e:
            logger.warning("工作经历保存跳过: %s", e)

    # 保存项目经历（追加模式）
    for proj_dict in req.projects:
        try:
            proj = Project(**proj_dict)
            store.add_project(proj)
            saved.setdefault("projects", []).append(proj.model_dump())
        except Exception as e:
            logger.warning("项目经历保存跳过: %s", e)

    # 保存证书（追加模式）
    for cert_dict in req.certificates:
        try:
            cert = Certificate(**cert_dict)
            store.add_certificate(cert)
            saved.setdefault("certificates", []).append(cert.model_dump())
        except Exception as e:
            logger.warning("证书保存跳过: %s", e)

    # 记录修改信息
    extraction_meta = {
        "source": "ai_extracted",
        "source_file": req.source_file,
        "modified_fields": req.modified_fields,
    }
    store.set_metadata("last_extraction", extraction_meta)

    return {
        "message": "提取结果已入库",
        "saved": saved,
        "modified_fields": req.modified_fields,
        "source": "ai_extracted" if not req.modified_fields else "ai_extracted + manual_corrections",
    }


# ── 简单上传（自动解析+保存） ──────────────────────


@router.post("/upload", summary="上传简历文件并直接入库（跳过预览）")
async def upload_resume(
    file: UploadFile = File(..., description="简历文件（PDF/DOCX）"),
):
    """上传简历文件，AI 提取后直接入库（跳过预览确认步骤）。

    如需预览后再确认，请使用 POST /upload/extract + POST /upload/confirm。
    """
    # 复用 extract 逻辑
    preview = await upload_and_extract(file)

    # 直接确认入库
    req = ConfirmSaveRequest(
        personal_info=preview.personal_info,
        education=preview.education,
        work_experience=preview.work_experience,
        projects=preview.projects,
        certificates=preview.certificates,
        modified_fields=[],
        source_file=file.filename or "",
    )
    return await confirm_and_save(req)


# ── 全量读取 ─────────────────────────────────────


@router.get("", summary="获取完整简历数据")
async def get_full_profile():
    """返回全部简历模块的完整数据。

    包括：personal_info, education, work_experience, projects, certificates。
    """
    store = get_profile_store()
    return store.get_all()


# ── 个人基本信息 ─────────────────────────────────


@router.put("/personal-info", summary="更新个人基本信息")
async def update_personal_info(
    info: PersonalInfo,
):
    """更新个人基本信息。传入完整 PersonalInfo 对象，全量覆盖。"""
    store = get_profile_store()
    updated = store.update_personal_info(info)
    return {"message": "个人基本信息已更新", "data": updated}


# ── 教育经历 ─────────────────────────────────────


@router.post("/education", summary="添加教育经历", status_code=status.HTTP_201_CREATED)
async def add_education(edu: Education):
    """添加一条教育经历。"""
    store = get_profile_store()
    added = store.add_education(edu)
    return {"message": "教育经历已添加", "data": added}


@router.put("/education/{index}", summary="更新指定教育经历")
async def update_education(index: int, edu: Education):
    """更新指定索引（0-based）的教育经历。索引超出范围返回 404。"""
    store = get_profile_store()
    try:
        updated = store.update_education(index, edu)
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"message": "教育经历已更新", "data": updated}


@router.delete("/education/{index}", summary="删除指定教育经历")
async def remove_education(index: int):
    """删除指定索引（0-based）的教育经历。索引超出范围返回 404。"""
    store = get_profile_store()
    try:
        store.remove_education(index)
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"message": "教育经历已删除"}


# ── 工作经历 ─────────────────────────────────────


@router.post(
    "/work-experience",
    summary="添加工作经历",
    status_code=status.HTTP_201_CREATED,
)
async def add_work_experience(exp: WorkExperience):
    """添加一条工作经历。"""
    store = get_profile_store()
    added = store.add_work_experience(exp)
    return {"message": "工作经历已添加", "data": added}


# ── 项目经历 ─────────────────────────────────────


@router.post("/project", summary="添加项目经历", status_code=status.HTTP_201_CREATED)
async def add_project(proj: Project):
    """添加一条项目经历。

    如果未传入项目 ID，系统会自动生成 UUID。
    """
    store = get_profile_store()
    added = store.add_project(proj)
    return {"message": "项目经历已添加", "data": added}


# ── 证书/资质 ────────────────────────────────────


@router.post(
    "/certificate",
    summary="添加证书/资质",
    status_code=status.HTTP_201_CREATED,
)
async def add_certificate(cert: Certificate):
    """添加一条证书或资质证明。"""
    store = get_profile_store()
    added = store.add_certificate(cert)
    return {"message": "证书/资质已添加", "data": added}
