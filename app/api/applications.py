"""智能招聘助手 — 投递记录管理 API

路由前缀: /api/applications

提供投递记录的完整 CRUD 以及统计和待跟进查询功能。
数据使用 ApplicationStore（SQLite 持久化）。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.models.schema import Application, ApplicationStatus
from app.storage.application_store import ApplicationStore

router = APIRouter(prefix="/api/applications", tags=["投递记录管理"])


# ── 依赖注入 ─────────────────────────────────────


def get_store() -> ApplicationStore:
    """获取 ApplicationStore 单例。"""
    from app.storage.application_store import application_store

    return application_store


# ── 请求/响应模型 ────────────────────────────────


from pydantic import BaseModel, Field


class ApplicationCreate(BaseModel):
    """创建投递记录请求"""

    company: str = Field(..., description="公司名称")
    title: str = Field(..., description="岗位名称")
    location: str = Field(..., description="工作地点")
    channel: str = Field(..., description="投递渠道，如 '国聘网', 'BOSS直聘'")
    url: str = Field(default="", description="投递链接")
    jd_text: Optional[str] = Field(None, description="JD 全文")
    resume_version: Optional[str] = Field(None, description="使用的简历版本")
    submit_date: Optional[str] = Field(None, description="投递日期，默认今天")
    notes: Optional[str] = Field(None, description="备注")


class ApplicationUpdate(BaseModel):
    """更新投递记录请求（所有字段可选，仅更新提供的内容）"""

    company: Optional[str] = Field(None, description="公司名称")
    title: Optional[str] = Field(None, description="岗位名称")
    location: Optional[str] = Field(None, description="工作地点")
    channel: Optional[str] = Field(None, description="投递渠道")
    url: Optional[str] = Field(None, description="投递链接")
    jd_text: Optional[str] = Field(None, description="JD 全文")
    resume_version: Optional[str] = Field(None, description="简历版本")
    cover_letter: Optional[str] = Field(None, description="求职信")
    submit_date: Optional[str] = Field(None, description="投递日期")
    status: Optional[str] = Field(None, description="当前状态")
    last_check: Optional[str] = Field(None, description="上次检查日期")
    next_check: Optional[str] = Field(None, description="下次检查日期")
    notes: Optional[str] = Field(None, description="备注")


class StatusUpdateRequest(BaseModel):
    """状态更新请求"""

    status: str = Field(..., description="新状态，如 '面试中', '已拒绝'")
    note: Optional[str] = Field(None, description="状态变更备注")


# ── 列表 ─────────────────────────────────────────


@router.get("", summary="获取投递记录列表")
async def list_applications(
    status: Optional[str] = Query(None, description="按状态筛选，如 '面试中'"),
):
    """获取所有投递记录列表，支持按状态过滤。结果按投递日期倒序排列。"""
    store = get_store()
    apps = store.list(filter_status=status)
    return {"total": len(apps), "applications": apps}


# ── 创建 ─────────────────────────────────────────


@router.post(
    "",
    summary="添加投递记录",
    status_code=status.HTTP_201_CREATED,
)
async def create_application(req: ApplicationCreate):
    """添加一条新的投递记录。

    如果未提供 submit_date，默认使用当天日期。
    """
    from datetime import date

    store = get_store()
    app = Application(
        company=req.company,
        title=req.title,
        location=req.location,
        url=req.url,
        jd_text=req.jd_text or "",
        channel=req.channel,
        resume_version=req.resume_version,
        submit_date=(
            date.fromisoformat(req.submit_date)
            if req.submit_date
            else date.today()
        ),
        notes=req.notes,
    )
    created = store.add(app)
    return {"message": "投递记录已创建", "data": created}


# ── 统计 ─────────────────────────────────────────


@router.get("/stats", summary="获取投递统计")
async def get_stats():
    """获取投递统计信息。

    包括：按状态分布数量、总数、待跟进数量。
    """
    store = get_store()
    return store.stats()


# ── 待跟进 ───────────────────────────────────────


@router.get("/pending-check", summary="获取待跟进投递记录")
async def get_pending_check():
    """获取需要跟进检查的投递记录。

    筛选条件：next_check 不为空且距离今天不超过 5 天。
    """
    store = get_store()
    apps = store.get_pending_check(days=5)
    return {"total": len(apps), "applications": apps}


# ── 单条查询 ─────────────────────────────────────


@router.get("/{id}", summary="获取单条投递记录")
async def get_application(id: str):
    """根据 ID 获取单条投递记录的详细信息。"""
    store = get_store()
    app = store.get(id)
    if app is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"投递记录不存在: {id}",
        )
    return app


# ── 更新 ─────────────────────────────────────────


@router.put("/{id}", summary="更新投递记录")
async def update_application(id: str, req: ApplicationUpdate):
    """更新投递记录（部分更新，仅覆盖提供的字段）。"""
    store = get_store()
    existing = store.get(id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"投递记录不存在: {id}",
        )

    from datetime import date

    update_data = req.model_dump(exclude_none=True)
    for field, value in update_data.items():
        if field == "submit_date" and isinstance(value, str):
            setattr(existing, field, date.fromisoformat(value))
        elif field == "last_check" and isinstance(value, str):
            setattr(existing, field, date.fromisoformat(value))
        elif field == "next_check" and isinstance(value, str):
            setattr(existing, field, date.fromisoformat(value))
        elif field == "status" and isinstance(value, str):
            existing.status = ApplicationStatus(value)
        else:
            setattr(existing, field, value)

    updated = store.update(existing)
    return {"message": "投递记录已更新", "data": updated}


# ── 状态更新 ─────────────────────────────────────


@router.put("/{id}/status", summary="更新投递状态")
async def update_application_status(id: str, req: StatusUpdateRequest):
    """仅更新投递记录的状态，并自动追加时间线记录。

    Args:
        id: 投递记录 ID
        req.status: 新状态（中文枚举值）
        req.note: 可选的状态变更备注
    """
    store = get_store()
    try:
        status_enum = ApplicationStatus(req.status)
    except ValueError:
        valid = [s.value for s in ApplicationStatus]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"无效的状态值: '{req.status}'。有效值: {valid}",
        )

    try:
        updated = store.update_status(id, status_enum, note=req.note or "")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return {"message": "状态已更新", "data": updated}


# ── 删除 ─────────────────────────────────────────


@router.delete("/{id}", summary="删除投递记录")
async def delete_application(id: str):
    """删除指定 ID 的投递记录。"""
    store = get_store()
    existing = store.get(id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"投递记录不存在: {id}",
        )
    store.delete(id)
    return {"message": "投递记录已删除", "id": id}
