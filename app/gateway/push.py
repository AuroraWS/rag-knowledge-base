"""WeChat Bot 主动推送管理 — 定时任务触发 → 模板消息。

5 种推送类型：
1. 今日待办（每日 9:00）
2. 每日复习（可配时间）
3. 面试倒计时（面试前 7 天起每日）
4. 状态变更通知（实时）
5. 每日日志（每日 22:00）
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from app.config import settings
from app.models.schema import ApplicationStatus
from app.storage.application_store import ApplicationStore, application_store

logger = logging.getLogger(__name__)


class PushManager:
    """推送管理器 — 构建推送内容 + 发送。"""

    def __init__(self, store: Optional[ApplicationStore] = None):
        self._store = store or application_store

    # ── 推送类型 1: 今日待办 ─────────────────────

    def build_daily_todo(self) -> str:
        """构建今日待办推送内容。"""
        from app.services.tracking_service import tracking_service

        todo = tracking_service.get_today_todo()
        summary = tracking_service.get_status_summary()

        lines = ["☀️ 早上好！今日求职待办", ""]
        lines.append("📊 投递概况")
        lines.append(f"  总投递: {summary.get('total', 0)}")
        lines.append(f"  面试中: {summary.get('interview_count', 0)}")
        lines.append(f"  待检查: {summary.get('pending_check_count', 0)}")

        if todo:
            lines.append("")
            lines.append(f"📋 今日待办 ({len(todo)} 项)")
            for item in todo:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    item.get("priority", "low"), "⚪"
                )
                lines.append(
                    f"  {icon} {item.get('company', '')} - {item.get('title', '')}"
                )
        else:
            lines.append("")
            lines.append("✅ 今日没有待办事项")

        return "\n".join(lines)

    # ── 推送类型 2: 面试倒计时 ───────────────────

    def build_interview_countdown(self) -> list[str]:
        """为每个即将面试的投递生成倒计时消息。"""
        messages = []
        today = date.today()
        apps = self._store.list()

        for app in apps:
            if app.interview_date and app.status == ApplicationStatus.INTERVIEWING:
                days_left = (app.interview_date - today).days
                if 0 <= days_left <= 7:
                    messages.append(
                        f"⏰ 面试倒计时: {app.company} - {app.title}\n"
                        f"   日期: {app.interview_date}\n"
                        f"   还剩 {days_left} 天，加油准备！"
                    )
        return messages

    # ── 推送类型 3: 每日日志 ────────────────────

    def build_daily_log(self) -> str:
        """构建每日日志推送内容。"""
        from app.services.tracking_service import tracking_service

        today = date.today()
        recent = tracking_service.get_recent_applications(days=1)
        summary = tracking_service.get_status_summary()

        lines = [f"📝 求职日报 ({today.isoformat()})", ""]

        if recent:
            lines.append(f"今日投递变动 ({len(recent)} 条):")
            for app in recent:
                lines.append(f"  • {app.company} - {app.title} ({app.status.value})")
        else:
            lines.append("今日无投递变动")

        lines.append("")
        lines.append("📊 状态分布:")
        for status, count in summary.get("by_status", {}).items():
            lines.append(f"  {status}: {count}")

        pending = self._store.get_pending_check(days=3)
        if pending:
            lines.append("")
            lines.append("📅 近期提醒:")
            for app in pending[:5]:
                lines.append(f"  • {app.company}: {app.next_action or '检查状态'}")

        return "\n".join(lines)

    # ── 发送 ────────────────────────────────────

    async def send(self, to_user: str, content: str) -> bool:
        """发送消息给指定用户。

        TODO: 接入真实微信模板消息 API 时替换此处实现。
        """
        if not settings.wechat_appid:
            logger.info("[WeChat Push 占位] to=%s, len=%d", to_user, len(content))
            return True

        logger.info("WeChat push sent to %s: %d chars", to_user, len(content))
        return True


# 模块级实例
push_manager = PushManager()
