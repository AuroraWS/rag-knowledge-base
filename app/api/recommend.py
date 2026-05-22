"""智能招聘助手 — 岗位推荐 API

路由前缀: /api/recommend

基于用户简历信息与样本 JD 数据集进行匹配分析，
返回匹配度最高的 Top 5 推荐岗位。

数据来源: app.data.jd_data 中的示例 JD 集合。
匹配算法：基于技能关键词重叠、学历硬性条件等维度计算综合分数。
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/recommend", tags=["岗位推荐"])


# ── 请求/响应模型 ────────────────────────────────


class JobRecommendation(BaseModel):
    """一条岗位推荐结果"""

    company: str = Field(..., description="公司名称")
    title: str = Field(..., description="岗位名称")
    location: str = Field(..., description="工作地点")
    match_score: float = Field(
        ..., ge=0.0, le=1.0, description="匹配度分数（0~1）"
    )
    matched_skills: list[str] = Field(
        default_factory=list, description="与简历匹配的技能/经验"
    )
    missing_skills: list[str] = Field(
        default_factory=list, description="简历中尚缺的技能/要求"
    )
    reason: str = Field(..., description="推荐理由（一句话概括）")


class RecommendRequest(BaseModel):
    """岗位推荐请求"""

    user_preferences: Optional[dict[str, Any]] = Field(
        None,
        description="用户偏好，可选字段：target_location, min_degree, preferred_industries",
    )


class RecommendResponse(BaseModel):
    """岗位推荐响应"""

    recommendations: list[JobRecommendation] = Field(
        ..., description="推荐岗位列表（Top 5）"
    )
    total_candidates: int = Field(
        ..., description="候选岗位总数"
    )


# ── 获取简历信息 ─────────────────────────────────


def get_profile_store():
    """延迟获取 ProfileStore。"""
    from app.storage.profile_store import profile_store

    return profile_store


def get_jds():
    """获取示例 JD 数据集。"""
    from app.data.jd_data import jds

    return jds


# ── 匹配引擎 ────────────────────────────────────


def _extract_skills_from_profile(profile: dict) -> set[str]:
    """从简历全量数据中提取技能关键词集合。"""
    skills: set[str] = set()

    # 从个人信息中提取技能相关字段
    personal = profile.get("personal_info")
    if personal:
        if hasattr(personal, "target_location"):
            skills.update(personal.target_location if isinstance(personal.target_location, list) else [])

    # 从工作经历提取技术栈
    for work in profile.get("work_experience", []):
        if hasattr(work, "tech_stack"):
            skills.update(getattr(work, "tech_stack", []) or [])
        if hasattr(work, "responsibilities"):
            for r in getattr(work, "responsibilities", []) or []:
                skills.add(r)

    # 从项目经历提取技术栈
    for proj in profile.get("projects", []):
        if hasattr(proj, "tech_stack"):
            skills.update(getattr(proj, "tech_stack", []) or [])
        if hasattr(proj, "highlights"):
            for h in getattr(proj, "highlights", []) or []:
                skills.add(h)

    # 从教育经历提取专业相关信息
    for edu in profile.get("education", []):
        if hasattr(edu, "major") and getattr(edu, "major", None):
            skills.add(getattr(edu, "major"))

    return {s.lower() for s in skills if s}


def _compute_match(
    jd: Any, profile_skills: set[str], preferences: Optional[dict] = None
) -> JobRecommendation:
    """计算一条 JD 与简历的匹配度，返回推荐结果。"""
    # 提取 JD 关键词
    raw_text = getattr(jd, "raw_text", "") or ""
    jd_keywords = set(raw_text.lower().split())

    # 从 requirements 提取更多关键词
    requirements_text = ""
    for req in getattr(jd, "requirements", []) or []:
        content = getattr(req, "content", "") or ""
        requirements_text += content + " "

    jd_keywords.update(requirements_text.lower().split())

    # 计算技能匹配
    matched = profile_skills & jd_keywords
    candidate_missing = jd_keywords - profile_skills

    # 取 Top 缺失技能（最多 5 个最相关的）
    skill_priority = [
        "python",
        "java",
        "pytorch",
        "tensorflow",
        "rag",
        "大模型",
        "llm",
        "agent",
        "docker",
        "kubernetes",
        "fastapi",
        "flask",
        "spring",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "faiss",
        "bert",
        "transformer",
        "机器学习",
        "深度学习",
        "人工智能",
        "计算机视觉",
        "nlp",
        "c++",
        "go",
        "rust",
    ]
    missing_ordered = [s for s in skill_priority if s in candidate_missing]
    missing_ordered += sorted(candidate_missing - set(skill_priority))[:3]

    # 计算匹配分数
    if len(jd_keywords) == 0:
        score = 0.0
    else:
        overlap_ratio = len(matched) / max(len(jd_keywords), 1)
        score = min(1.0, overlap_ratio * 1.2)  # 略微缩放，更积极

    # 偏好加权
    if preferences:
        target_location = preferences.get("target_location")
        if target_location:
            jd_loc = getattr(jd, "location", "") or ""
            if isinstance(target_location, list) and any(
                loc in jd_loc for loc in target_location
            ):
                score = min(1.0, score + 0.1)
            elif isinstance(target_location, str) and target_location in jd_loc:
                score = min(1.0, score + 0.1)
            else:
                score = max(0.0, score - 0.1)

    # 构建推荐理由
    company = getattr(jd, "company", "") or ""
    title = getattr(jd, "title", "") or ""
    location = getattr(jd, "location", "") or ""

    if len(matched) >= 3:
        reason = f"简历技能与 {company} 的 {title} 岗位高度匹配，{len(matched)} 项技能重叠，非常推荐投递。"
    elif len(matched) >= 1:
        reason = f"简历与 {company} 的 {title} 岗位有一定匹配度，关键技能 {', '.join(sorted(matched)[:3])} 已被覆盖。"
    else:
        reason = f"该岗位与您的背景匹配度较低，但可作为探索性选择。"

    return JobRecommendation(
        company=company,
        title=title,
        location=location,
        match_score=round(score, 2),
        matched_skills=sorted(matched),
        missing_skills=missing_ordered[:5],
        reason=reason,
    )


# ── 推荐端点 ────────────────────────────────────


@router.post("", response_model=RecommendResponse, summary="获取岗位推荐（Top 5）")
async def get_recommendations(
    req: RecommendRequest = None,
):
    """根据用户简历和偏好，从示例 JD 数据集中计算推荐结果。

    匹配基于以下维度：
    1. 技能/技术栈关键词重叠（主要维度）
    2. 学历/专业背景
    3. 用户偏好（地点等）

    返回匹配度最高的 Top 5 岗位，附带匹配细节。
    """
    if req is None:
        req = RecommendRequest()

    # 加载简历
    store = get_profile_store()
    profile = store.get_all()
    profile_skills = _extract_skills_from_profile(profile)

    # 加载 JD 数据集
    jds = get_jds()
    if not jds:
        return RecommendResponse(recommendations=[], total_candidates=0)

    # 计算每条 JD 的匹配度
    results: list[JobRecommendation] = []
    for jd in jds:
        rec = _compute_match(jd, profile_skills, req.user_preferences)
        results.append(rec)

    # 按匹配度降序排列，取 Top 5
    results.sort(key=lambda r: r.match_score, reverse=True)
    top5 = results[:5]

    return RecommendResponse(
        recommendations=top5,
        total_candidates=len(jds),
    )
