"""智能招聘助手 — 简历信息管理 API

路由前缀: /api/profile

提供简历五个模块的完整 CRUD：
- 个人基本信息 (PersonalInfo)
- 教育经历 (Education)
- 工作经历 (WorkExperience)
- 项目经历 (Project)
- 证书/资质 (Certificate)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schema import (
    Certificate,
    Education,
    PersonalInfo,
    Project,
    WorkExperience,
)
from app.storage.profile_store import ProfileStore

router = APIRouter(prefix="/api/profile", tags=["简历信息管理"])


# ── 依赖注入 ─────────────────────────────────────


def get_profile_store() -> ProfileStore:
    """获取 ProfileStore 单例。"""
    from app.storage.profile_store import profile_store

    return profile_store


# ── 简历上传 ─────────────────────────────────────


@router.post("/upload", summary="上传简历文件并提取信息")
async def upload_resume(
    file: UploadFile = File(..., description="简历文件（PDF/Word/图片等）"),
):
    """上传简历文件，返回提取的简历信息摘要。

    当前为接口存根：接收文件并返回基本信息，
    后续可接入文档解析 + LLM 提取管线。
    """
    content = await file.read()
    file_size = len(content)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": file_size,
        "message": "文件已接收，简历解析管线尚未接入。返回占位信息。",
        "extracted_info": {
            "name": "（待解析）",
            "email": "（待解析）",
            "phone": "（待解析）",
        },
    }


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
