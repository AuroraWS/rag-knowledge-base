"""智能招聘助手 — 自然语言命令解析 API

路由前缀: /api/command

接收用户自然语言输入（中文），通过 LLM 解析意图，
自动执行对应操作（投递记录管理、生成内容等）。

支持意图（intent）：
- add_application:     添加投递记录
- update_status:       更新投递状态
- list_applications:   查询投递列表
- generate_self_intro: 生成自我介绍
- query_profile:       查询简历信息
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.schema import Application, ApplicationStatus
from app.rag.generator import LLMGenerator
from app.storage.application_store import ApplicationStore

router = APIRouter(prefix="/api/command", tags=["自然语言命令"])


# ── 依赖注入 ─────────────────────────────────────


def get_generator() -> LLMGenerator:
    """获取 LLMGenerator 实例。"""
    return LLMGenerator()


def get_application_store() -> ApplicationStore:
    """获取 ApplicationStore 单例。"""
    from app.storage.application_store import application_store

    return application_store


def get_profile_store():
    """获取 ProfileStore 单例。"""
    from app.storage.profile_store import profile_store

    return profile_store


# ── 请求/响应模型 ────────────────────────────────


class CommandRequest(BaseModel):
    """自然语言命令请求"""

    text: str = Field(..., description="用户的自然语言输入，如 '我投了JPMorgan的Software Engineer'")


class CommandResponse(BaseModel):
    """命令解析与执行结果"""

    intent: str = Field(..., description="识别的意图类型")
    original_text: str = Field(..., description="原始输入文本")
    action_summary: str = Field(..., description="执行摘要")
    result: Any = Field(None, description="执行结果数据")


# ── 意图解析 System Prompt ──────────────────────

_INTENT_SYSTEM_PROMPT = """你是一个智能招聘助手的意图识别引擎。

你的任务是将用户的中文自然语言输入解析为一个结构化的意图对象。
用户输入的可能是：
1. 投递记录相关的操作（添加、更新状态、查询列表）
2. 生成内容的请求（生成自我介绍）
3. 查询简历信息的请求

请严格按以下 JSON 格式输出，不要添加其他内容：

对于"添加投递记录"（add_application）：
{
    "intent": "add_application",
    "confidence": 0.95,
    "params": {
        "company": "公司名称",
        "title": "岗位名称",
        "location": "工作地点（如未知则填空字符串）",
        "channel": "投递渠道（如未知则填'手动录入'）",
        "url": "",
        "jd_text": ""
    }
}

对于"更新状态"（update_status）：
{
    "intent": "update_status",
    "confidence": 0.95,
    "params": {
        "company": "公司名称（部分匹配用）",
        "title": "岗位名称（可选）",
        "new_status": "新状态（必须为：待投递/已投递待反馈/已收到笔试/面试中/已拒绝/已拿到Offer）",
        "note": "备注（可选）"
    }
}

对于"查询投递列表"（list_applications）：
{
    "intent": "list_applications",
    "confidence": 0.95,
    "params": {
        "filter_status": "可选的状态过滤（不限制则为null）"
    }
}

对于"生成自我介绍"（generate_self_intro）：
{
    "intent": "generate_self_intro",
    "confidence": 0.95,
    "params": {
        "jd_text": "JD内容（从用户输入中提取）"
    }
}

对于"查询简历"（query_profile）：
{
    "intent": "query_profile",
    "confidence": 0.95,
    "params": {}
}

如果无法识别意图：
{
    "intent": "unknown",
    "confidence": 0.0,
    "params": {},
    "error": "无法理解您的指令，请重新描述"
}

当前日期：{today}"""


# ── 核心端点 ─────────────────────────────────────


@router.post("", response_model=CommandResponse, summary="解析并执行自然语言命令")
async def execute_command(req: CommandRequest):
    """接收自然语言输入，自动识别意图并执行对应操作。

    支持的操作：
    - 添加投递记录："我投了JPMorgan的Software Engineer"
    - 更新状态："字节跳动三面通过了"
    - 查询投递："我的投递记录有哪些" / "面试中的有哪些"
    - 生成自我介绍："帮我写一份JPMorgan的自我介绍"
    - 查询简历："我的简历信息"
    """
    generator = get_generator()

    # 1. LLM 意图解析
    try:
        parsed = await generator.generate_structured(
            prompt=f"用户输入：{req.text}\n\n请解析用户意图并返回 JSON。",
            system_prompt=_INTENT_SYSTEM_PROMPT.format(today=date.today().isoformat()),
            temperature=0.1,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"意图解析失败: {str(e)}",
        )

    if isinstance(parsed, dict):
        intent_data = parsed
    else:
        # 如果是 Pydantic model
        intent_data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)

    intent = intent_data.get("intent", "unknown")
    params = intent_data.get("params", {})

    # 2. 分发执行
    try:
        if intent == "add_application":
            return await _handle_add_application(params, req.text)
        elif intent == "update_status":
            return await _handle_update_status(params, req.text)
        elif intent == "list_applications":
            return await _handle_list_applications(params, req.text)
        elif intent == "generate_self_intro":
            return await _handle_generate_self_intro(params, req.text)
        elif intent == "query_profile":
            return await _handle_query_profile(req.text)
        else:
            return CommandResponse(
                intent="unknown",
                original_text=req.text,
                action_summary=intent_data.get("error", "无法理解您的指令"),
                result=None,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"命令执行失败: {str(e)}",
        )


# ── 意图处理器 ───────────────────────────────────


async def _handle_add_application(params: dict, original: str) -> CommandResponse:
    """处理添加投递记录意图。"""
    store = get_application_store()

    app = Application(
        company=params.get("company", ""),
        title=params.get("title", ""),
        location=params.get("location", ""),
        url=params.get("url", ""),
        channel=params.get("channel", "手动录入"),
        jd_text=params.get("jd_text", ""),
        submit_date=date.today(),
    )
    created = store.add(app)

    return CommandResponse(
        intent="add_application",
        original_text=original,
        action_summary=f"已添加投递记录：{created.company} - {created.title}",
        result=created.model_dump(),
    )


async def _handle_update_status(params: dict, original: str) -> CommandResponse:
    """处理更新状态意图。"""
    store = get_application_store()
    company = params.get("company", "")
    new_status_str = params.get("new_status", "")
    note = params.get("note", "")

    # 查找匹配的公司（使用模糊匹配）
    all_apps = store.list()
    target = None
    for app in all_apps:
        if company.lower() in app.company.lower():
            target = app
            break

    if target is None:
        return CommandResponse(
            intent="update_status",
            original_text=original,
            action_summary=f"未找到与 '{company}' 相关的投递记录",
            result=None,
        )

    try:
        new_status = ApplicationStatus(new_status_str)
    except ValueError:
        valid = [s.value for s in ApplicationStatus]
        return CommandResponse(
            intent="update_status",
            original_text=original,
            action_summary=f"无效状态 '{new_status_str}'，有效值: {valid}",
            result=None,
        )

    updated = store.update_status(target.id, new_status, note=note)
    return CommandResponse(
        intent="update_status",
        original_text=original,
        action_summary=f"已将 {updated.company} - {updated.title} 的状态更新为: {new_status_str}",
        result={"id": updated.id, "company": updated.company, "title": updated.title, "status": updated.status.value},
    )


async def _handle_list_applications(params: dict, original: str) -> CommandResponse:
    """处理查询投递列表意图。"""
    store = get_application_store()
    filter_status = params.get("filter_status")
    apps = store.list(filter_status=filter_status or None)

    return CommandResponse(
        intent="list_applications",
        original_text=original,
        action_summary=f"共找到 {len(apps)} 条投递记录" + (f"（状态: {filter_status}）" if filter_status else ""),
        result=[a.model_dump() for a in apps],
    )


async def _handle_generate_self_intro(params: dict, original: str) -> CommandResponse:
    """处理生成自我介绍意图。"""
    generator = get_generator()
    store = get_profile_store()
    profile = store.get_all()
    jd_text = params.get("jd_text", "")

    system_prompt = "你是一个专业的求职面试助手。请根据候选人的简历和目标岗位生成有针对性的自我介绍。"
    prompt = f"""请为候选人撰写一段完整的自我介绍（200-300字）。

【候选简历】
{profile}

【目标岗位描述】
{jd_text or '从用户输入中提取的岗位描述'}

请直接输出自我介绍文本。"""

    text = await generator.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.3,
    )

    return CommandResponse(
        intent="generate_self_intro",
        original_text=original,
        action_summary="自我介绍已生成",
        result={"intro": text},
    )


async def _handle_query_profile(original: str) -> CommandResponse:
    """处理查询简历信息意图。"""
    store = get_profile_store()
    profile = store.get_all()

    personal = profile.get("personal_info")
    name = personal.name if personal and hasattr(personal, "name") else "未填写"

    return CommandResponse(
        intent="query_profile",
        original_text=original,
        action_summary=f"已查询 {name} 的简历信息",
        result=profile,
    )
