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

    # 从工作经历提取技术栈
    for work in profile.get("work_experience", []):
        tech = getattr(work, "tech_stack", None)
        if tech:
            skills.update(str(t).lower() for t in tech)

    # 从项目经历提取技术栈
    for proj in profile.get("projects", []):
        tech = getattr(proj, "tech_stack", None)
        if tech:
            skills.update(str(t).lower() for t in tech)

    # 从教育经历提取专业 + 学位
    for edu in profile.get("education", []):
        major = getattr(edu, "major", None)
        if major:
            skills.add(str(major).lower())
        degree = getattr(edu, "degree", None)
        if degree:
            skills.add(str(degree).lower())

    return {s for s in skills if s and len(s) >= 2}


def _compute_match(
    jd: Any, profile_skills: set[str], preferences: Optional[dict] = None
) -> JobRecommendation:
    """计算一条 JD 与简历的匹配度。

    使用子串匹配代替集合交集——因为中文文本无空格分隔，
    .split() 会把整段文本当一个元素，导致交集永远为空。
    """
    raw_text = (getattr(jd, "raw_text", "") or "").lower()

    # 拼合所有 JD 可检索文本（raw_text + requirements）
    jd_search_text = raw_text
    for req in getattr(jd, "requirements", []) or []:
        content = req if isinstance(req, str) else (getattr(req, "content", "") or "")
        jd_search_text += " " + content.lower()

    # 子串匹配：每项 Profile 技能是否在 JD 文本中出现
    matched = {skill for skill in profile_skills if skill.lower() in jd_search_text}

    # JD 要求清单（结构化字段）
    jd_requirements: list[str] = []
    for req in getattr(jd, "requirements", []) or []:
        if isinstance(req, str):
            jd_requirements.append(req)
        elif hasattr(req, "content"):
            jd_requirements.append(getattr(req, "content", ""))

    # 缺失 = JD 要求中未被 Profile 技能子串匹配到的
    missing = [req for req in jd_requirements if not any(
        req.lower() in skill.lower() or skill.lower() in req.lower()
        for skill in profile_skills
    )]

    # 匹配分数：基于 JD 要求命中率
    if jd_requirements:
        score = len(matched) / max(len(jd_requirements), 1)
        score = min(1.0, score * 1.2)
    elif profile_skills:
        score = min(0.5, len(matched) * 0.1)
    else:
        score = 0.0

    # 偏好加权
    if preferences:
        target_location = preferences.get("target_location")
        if target_location:
            jd_loc = (getattr(jd, "location", "") or "").lower()
            if isinstance(target_location, list) and any(
                loc.lower() in jd_loc for loc in target_location
            ):
                score = min(1.0, score + 0.1)
            elif isinstance(target_location, str) and target_location.lower() in jd_loc:
                score = min(1.0, score + 0.1)
            else:
                score = max(0.0, score - 0.05)

    # 构建推荐理由
    company = getattr(jd, "company", "") or ""
    title = getattr(jd, "title", "") or ""
    location = getattr(jd, "location", "") or ""

    if len(matched) >= 3:
        top3 = sorted(matched)[:3]
        reason = f"简历技能与 {company} 的 {title} 岗位高度匹配，{len(matched)} 项技能重叠（{', '.join(top3)}等），非常推荐投递。"
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
        missing_skills=missing[:5],
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
