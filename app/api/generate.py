"""智能招聘助手 — LLM 内容生成 API

路由前缀: /api/generate

基于 DeepSeek API（通过 LLMGenerator）生成三类求职相关内容：
- 自我介绍（自荐书）：支持短版/完整版/英文版
- 项目介绍：STAR 格式
- 求职信：按公司、岗位定制
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models.schema import Project
from app.rag.generator import LLMGenerator

router = APIRouter(prefix="/api/generate", tags=["LLM 内容生成"])


# ── 依赖注入 ─────────────────────────────────────


def get_generator() -> LLMGenerator:
    """获取 LLMGenerator 实例。"""
    from app.rag.generator import LLMGenerator as Gen

    return Gen()


def get_profile_store():
    """获取 ProfileStore 实例（延迟导入避免循环依赖）。"""
    from app.storage.profile_store import profile_store

    return profile_store


# ── 请求/响应模型 ────────────────────────────────


from pydantic import BaseModel, Field


class SelfIntroRequest(BaseModel):
    """自我介绍生成请求"""

    jd_text: str = Field(..., description="JD 全文，用于匹配个人经历生成有针对性的自荐")
    style: Optional[str] = Field(
        None,
        description="风格偏好，如 'formal'（正式）, 'passionate'（热情）, 'concise'（简洁）",
    )
    focus_skills: Optional[list[str]] = Field(
        None, description="重点突出的技能列表"
    )


class SelfIntroResponse(BaseModel):
    """自我介绍生成响应"""

    short_version: str = Field(..., description="一句话版（30 字以内）")
    full_version: str = Field(..., description="完整版（200-300 字）")
    english_version: str = Field(..., description="英文版")


class ProjectIntroRequest(BaseModel):
    """项目介绍生成请求"""

    project_id: str = Field(..., description="项目 ID（UUID）")
    length: str = Field(
        default="medium",
        description="篇幅偏好: 'short'（50字）, 'medium'（150字）, 'long'（300字）",
    )


class CoverLetterRequest(BaseModel):
    """求职信生成请求"""

    company: str = Field(..., description="公司名称")
    position: str = Field(..., description="岗位名称")
    jd_text: str = Field(..., description="JD 全文")
    tone: Optional[str] = Field(
        None, description="语气偏好，如 'professional', 'warm', 'enthusiastic'"
    )


# ── 自荐信/自我介绍 ──────────────────────────────


@router.post(
    "/self-intro",
    response_model=SelfIntroResponse,
    summary="生成自我介绍（自荐书）",
)
async def generate_self_intro(req: SelfIntroRequest):
    """根据 JD 内容和简历数据生成三段式自我介绍。

    返回三个版本：
    - short_version：一句话版（30 字以内，适合简历开头）
    - full_version：完整版（200-300 字，适合自荐信正文）
    - english_version：英文版
    """
    store = get_profile_store()
    profile = store.get_all()
    personal = profile.get("personal_info", {})
    name = getattr(personal, "name", "候选人") if hasattr(personal, "name") else "候选人"

    style_note = f"\n风格要求：{req.style}" if req.style else ""
    focus_note = (
        f"\n重点突出技能：{', '.join(req.focus_skills)}"
        if req.focus_skills
        else ""
    )

    system_prompt = """你是一个专业的求职面试助手，帮助候选人根据目标岗位（JD）生成有针对性的自我介绍。
你熟悉 STAR 法则和中文求职场景的表达习惯。
严格按要求的格式输出，不添加额外说明。"""

    prompt = f"""请根据以下候选人的简历信息和目标岗位 JD，生成三段式自我介绍。

【候选简历信息】
{profile}

【目标岗位 JD】
{req.jd_text}{style_note}{focus_note}

请按以下 JSON 格式输出，不要添加其他内容：
{{
    "short_version": "一句话版自我介绍（30字以内）",
    "full_version": "完整版自我介绍（200-300字，突出与JD匹配的经历和技能）",
    "english_version": "English version (150-200 words)"
}}"""

    generator = get_generator()
    try:
        result = await generator.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        # 如果返回的是 dict 或 BaseModel，转为响应
        if isinstance(result, dict):
            return SelfIntroResponse(**result)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"自我介绍生成失败: {str(e)}",
        )


# ── 项目介绍（STAR 格式） ────────────────────────


@router.post("/project-intro", summary="生成项目介绍（STAR 格式）")
async def generate_project_intro(req: ProjectIntroRequest):
    """根据项目 ID 从简历中查找项目经历，生成 STAR 格式的项目介绍。

    支持 short / medium / long 三种篇幅。
    """
    store = get_profile_store()
    projects = store.get_projects()
    target = None
    for p in projects:
        if p.id == req.project_id:
            target = p
            break

    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"未找到项目 ID: {req.project_id}",
        )

    length_guide = {
        "short": "50字以内，突出最核心的成果",
        "medium": "150字左右，包含 Situation、Task、Action、Result",
        "long": "300字左右，详尽的 STAR 描述",
    }
    length_desc = length_guide.get(req.length, length_guide["medium"])

    system_prompt = """你是一个专业的求职面试助手，擅长用 STAR 法则撰写项目经历介绍。
让项目描述既有技术深度又有可读性，适合在面试或简历中使用。"""

    prompt = f"""请将以下项目经历按照 STAR 法则整理成项目介绍。

【项目信息】
项目名称：{target.name}
角色：{target.role}
时间：{target.start_date} ~ {target.end_date or '至今'}
描述：{target.description}
技术栈：{', '.join(target.tech_stack)}
亮点/核心贡献：
{chr(10).join(f'- {h}' for h in target.highlights)}

【篇幅要求】
{length_desc}

请直接输出项目介绍文本，不要添加额外说明。"""

    generator = get_generator()
    try:
        text = await generator.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        return {
            "project_id": target.id,
            "project_name": target.name,
            "length": req.length,
            "intro": text,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"项目介绍生成失败: {str(e)}",
        )


# ── 求职信 ──────────────────────────────────────


@router.post("/cover-letter", summary="生成求职信")
async def generate_cover_letter(req: CoverLetterRequest):
    """根据公司、岗位和 JD 内容，生成定制化的求职信草稿。"""
    store = get_profile_store()
    profile = store.get_all()
    personal = profile.get("personal_info", {})
    name = getattr(personal, "name", "候选人") if hasattr(personal, "name") else "候选人"

    tone_note = ""
    tone_map = {
        "professional": "语气正式、专业，突出能力匹配",
        "warm": "语气亲切温和，体现对公司的认同",
        "enthusiastic": "语气热情积极，展示强烈的加入意愿",
    }
    if req.tone:
        tone_note = f"\n语气要求：{tone_map.get(req.tone, req.tone)}"

    system_prompt = """你是一个专业的求职信写作助手。请根据候选人简历和目标岗位信息，撰写一封得体的中文求职信。
求职信需包含：开头问候、自我介绍、能力匹配说明、对公司的认同、结尾致谢。
格式规范、措辞得体、长度约300-500字。"""

    prompt = f"""请为以下候选人撰写一封发往 {req.company} 的求职信。

【应聘岗位】{req.position}

【候选人简历】
{profile}

【JD 全文】
{req.jd_text}{tone_note}

请直接输出求职信正文（包含称呼、正文、落款），不要添加额外说明。"""

    generator = get_generator()
    try:
        text = await generator.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        return {
            "company": req.company,
            "position": req.position,
            "cover_letter": text,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"求职信生成失败: {str(e)}",
        )
