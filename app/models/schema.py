"""智能招聘助手 — Pydantic 数据模型

集中定义所有数据模型，供 API 层、存储层、服务层和 Agent 层共享使用。

模型分组：
- 枚举: ApplicationStatus
- 资料库模型: PersonalInfo, Education, WorkExperience, Project, Certificate
- 投递模型: Application, StatusChange
- 辅助模型: FieldMemory, MatchResult, HealthResponse
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════


class ApplicationStatus(StrEnum):
    """投递状态枚举。"""

    PENDING = "待投递"
    APPLIED = "已投递待反馈"
    EXAM = "已收到笔试"
    INTERVIEWING = "面试中"
    REJECTED = "已拒绝"
    OFFER = "已拿到Offer"


# ═══════════════════════════════════════════════════════
# 通用响应模型
# ═══════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """健康检查响应"""

    version: str


# ═══════════════════════════════════════════════════════
# 资料库模型
# ═══════════════════════════════════════════════════════


class PersonalInfo(BaseModel):
    """个人基本信息"""

    name: str = ""
    phone: str = ""
    email: str = ""
    wechat: Optional[str] = None
    target_location: list[str] = Field(default_factory=list)
    target_salary: Optional[str] = None
    earliest_start_date: Optional[str] = None
    id_number: Optional[str] = None
    birthday: Optional[str] = None
    gender: Optional[str] = None
    residence: Optional[str] = None
    source_files: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """教育经历"""

    school: str
    degree: str
    major: str
    start_date: str
    end_date: str
    gpa: Optional[float] = None
    degree_cert_number: Optional[str] = None
    is_overseas: bool = False
    source_file: Optional[str] = None


class WorkExperience(BaseModel):
    """工作经历"""

    company: str
    department: Optional[str] = None
    title: str
    start_date: str
    end_date: Optional[str] = None
    is_current: bool = False
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    source_file: Optional[str] = None


class Project(BaseModel):
    """项目经历"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    source_file: Optional[str] = None


class Certificate(BaseModel):
    """证书/资质"""

    name: str
    issuer: str
    date: Optional[str] = None
    cert_number: Optional[str] = None
    source_file: Optional[str] = None


# ═══════════════════════════════════════════════════════
# 投递模型
# ═══════════════════════════════════════════════════════


class StatusChange(BaseModel):
    """状态变更记录，嵌入 Application.timeline"""

    status: ApplicationStatus
    change_date: date = Field(alias="date")
    note: Optional[str] = None

    model_config = {"populate_by_name": True}


class Application(BaseModel):
    """投递记录 — 核心实体"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company: str
    title: str
    location: str = ""
    url: str = ""
    jd_text: str = ""
    channel: str = ""
    resume_version: Optional[str] = None
    cover_letter: Optional[str] = None
    submit_date: date = Field(default_factory=date.today)
    status: ApplicationStatus = Field(default=ApplicationStatus.APPLIED)
    last_check: Optional[date] = None
    next_check: Optional[date] = None
    timeline: list[StatusChange] = Field(default_factory=list)
    notes: str = ""
    # 面试 & 准备计划
    interview_date: Optional[date] = None
    prep_plan: Optional[dict[str, Any]] = None
    # 延伸字段
    job_id: Optional[str] = None
    jd_summary: Optional[str] = None
    next_action: Optional[str] = None
    next_action_date: Optional[date] = None


# ═══════════════════════════════════════════════════════
# 辅助模型
# ═══════════════════════════════════════════════════════


class FieldMemory(BaseModel):
    """字段记忆 — 系统记住用户填过的值"""

    field_key: str
    field_label: str
    value: str
    first_seen: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    source_context: Optional[str] = None
    confidence: float = 1.0


class MatchResult(BaseModel):
    """简历 vs JD 匹配分析结果"""

    resume_name: str = ""
    jd_title: str = ""
    company: str = ""
    match_score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    analysis: str = ""
    skill_matches: list[dict[str, Any]] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
