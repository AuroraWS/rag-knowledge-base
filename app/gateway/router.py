"""WeChat Bot 消息路由 — chatbot 对话 + 命令分发。

将微信消息路由到三大处理分支：
1. 求职命令 → /api/command（添加投递、更新状态等）
2. 求职问答 → 查询资料库 / 调 RAG 生成
3. 日常聊天 → LLM 自由对话
"""

from __future__ import annotations

import logging
from typing import Optional

from app.gateway.wechat import GatewayMessage
from app.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

# ── 命令关键词（触发命令引擎） ──────────────────

_COMMAND_KEYWORDS = [
    "投了", "投递", "面试", "笔试", "状态",
    "拒绝", "offer", "Offer", "已拿",
    "帮我看", "帮我查", "帮我准备", "准备",
    "列表", "统计", "仪表盘",
    "自我介绍", "生成",
]

_CHAT_SYSTEM_PROMPT = """你是王爽的求职助手，运行在微信上。
你的角色：
- 帮助管理投递记录
- 提醒面试和检查日期
- 回答求职相关问题
- 提供职业建议

保持回复简洁（微信消息不适合长篇大论），每条回复控制在200字以内。
如果用户说的是求职相关操作，引导他们使用明确的命令格式。

当前你的能力：
- 记录投递："投了[公司名]的[岗位名]" → 自动记录
- 查状态："投递列表" / "面试中的有哪些"
- 生成介绍："帮我写[公司名]的自我介绍"
"""


class MessageRouter:
    """消息路由器 — 判断意图并分发到对应处理器。"""

    def __init__(self):
        self._generator: Optional[LLMGenerator] = None

    @property
    def generator(self) -> LLMGenerator:
        if self._generator is None:
            self._generator = LLMGenerator()
        return self._generator

    # ── 意图判断 ────────────────────────────────

    def is_command(self, msg: GatewayMessage) -> bool:
        """判断是否为求职命令（触发命令引擎）。"""
        return any(kw in msg.content for kw in _COMMAND_KEYWORDS)

    def is_subscribe(self, msg: GatewayMessage) -> bool:
        """判断是否为订阅/退订指令。"""
        content = msg.content.strip()
        return content in ("订阅", "退订", "开启推送", "关闭推送")

    # ── 路由分发 ─────────────────────────────────

    async def route(self, msg: GatewayMessage) -> str:
        """路由消息到对应处理器，返回回复文本。"""
        if self.is_subscribe(msg):
            return self._handle_subscribe(msg)

        if self.is_command(msg):
            return await self._handle_command(msg)

        return await self._handle_chat(msg)

    # ── 处理器 ────────────────────────────────────

    def _handle_subscribe(self, msg: GatewayMessage) -> str:
        """处理订阅/退订。"""
        if "退订" in msg.content or "关闭" in msg.content:
            return "已关闭每日推送。需要时回复「订阅」重新开启。"
        return "已开启每日推送！每天 9:00 发送今日待办，22:00 发送每日日志。"

    async def _handle_command(self, msg: GatewayMessage) -> str:
        """转发到 /api/command 引擎。"""
        try:
            from app.api.command import CommandRequest, execute_command

            req = CommandRequest(text=msg.content)
            resp = await execute_command(req)
            return resp.action_summary
        except Exception as e:
            logger.error("命令执行失败: %s", e)
            return f"抱歉，处理指令时出错了: {e}"

    async def _handle_chat(self, msg: GatewayMessage) -> str:
        """LLM 自由对话。"""
        try:
            text = await self.generator.generate(
                prompt=msg.content,
                system_prompt=_CHAT_SYSTEM_PROMPT,
                temperature=0.7,
            )
            if len(text) > 400:
                text = text[:380] + "..."
            return text
        except Exception as e:
            logger.error("Chat 回复失败: %s", e)
            return "收到你的消息了，但我现在脑子有点转不动，请稍后再试～"


# 模块级实例
message_router = MessageRouter()
