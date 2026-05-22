"""投递跟踪服务 — 今日待办 / 到期检查 / 状态统计。

包装 ApplicationStore 并添加业务逻辑层：
- 计算今日需要处理的待办事项
- 筛选需要关注的投递记录
- 生成状态分布统计
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from app.models.schema import Application, ApplicationStatus
from app.storage.application_store import ApplicationStore, application_store as _store

logger = logging.getLogger(__name__)


class TrackingService:
    """投递跟踪服务 — 连接 ApplicationStore 的业务逻辑层。"""

    def __init__(self, store: Optional[ApplicationStore] = None) -> None:
        self._store = store or _store

    # ── 今日待办 ────────────────────────────────────

    def get_today_todo(self) -> list[dict]:
        """获取今日待办事项列表。

        Returns:
            [
                {
                    "type": "pending_check" | "upcoming_interview" | "pending_apply",
                    "application_id": "uuid",
                    "company": "公司名",
                    "title": "岗位",
                    "detail": "描述信息",
                    "priority": "high" | "medium" | "low",
                }
            ]
        """
        todo: list[dict] = []
        today = date.today()

        # 1. 需要跟进检查的记录
        pending = self._store.get_pending_check(days=3)
        for app in pending:
            priority = "high" if app.next_check and app.next_check <= today else "medium"
            todo.append({
                "type": "pending_check",
                "application_id": app.id,
                "company": app.company,
                "title": app.title,
                "detail": f"上次检查: {app.last_check.isoformat() if app.last_check else '从未'}, "
                          f"下次应检查: {app.next_check.isoformat() if app.next_check else '未设置'}",
                "priority": priority,
            })

        # 2. 面试中的记录（今日或近期有面试安排的）
        interviewing = self._store.list(filter_status=ApplicationStatus.INTERVIEWING.value)
        for app in interviewing:
            # 检查备注中是否包含面试日期信息
            notes = app.notes or ""
            if "面试" in notes or "面" in notes:
                todo.append({
                    "type": "upcoming_interview",
                    "application_id": app.id,
                    "company": app.company,
                    "title": app.title,
                    "detail": f"状态: 面试中 | {notes[:100]}",
                    "priority": "high",
                })
            else:
                todo.append({
                    "type": "upcoming_interview",
                    "application_id": app.id,
                    "company": app.company,
                    "title": app.title,
                    "detail": f"状态: 面试中，请留意面试安排",
                    "priority": "medium",
                })

        # 3. 待投递的记录
        pending_apply = self._store.list(filter_status=ApplicationStatus.PENDING.value)
        for app in pending_apply:
            todo.append({
                "type": "pending_apply",
                "application_id": app.id,
                "company": app.company,
                "title": app.title,
                "detail": f"渠道: {app.channel} | 链接: {app.url}",
                "priority": "medium",
            })

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        todo.sort(key=lambda x: priority_order.get(x["priority"], 99))

        return todo

    # ── 到期检查 ────────────────────────────────────

    def check_due_applications(self, days: int = 5) -> list[Application]:
        """获取需要关注的到期投递记录。

        Args:
            days: 提前关注的天数范围。

        Returns:
            需要关注的 Application 列表。
        """
        return self._store.get_pending_check(days=days)

    # ── 状态统计 ────────────────────────────────────

    def get_status_summary(self) -> dict:
        """获取投递状态分布统计。

        Returns:
            {
                "total": 总投递数,
                "by_status": { "待投递": n, "面试中": n, ... },
                "pending_check_count": 需要检查数,
                "interview_count": 面试中数量,
                "offer_count": 已拿Offer数量,
                "reject_count": 已拒绝数量,
            }
        """
        stats = self._store.stats()
        by_status = stats.get("by_status", {})

        return {
            "total": stats.get("total", 0),
            "by_status": by_status,
            "pending_check_count": stats.get("pending_check", 0),
            "interview_count": by_status.get(ApplicationStatus.INTERVIEWING.value, 0),
            "offer_count": by_status.get(ApplicationStatus.OFFER.value, 0),
            "reject_count": by_status.get(ApplicationStatus.REJECTED.value, 0),
        }

    # ── 快捷统计 ────────────────────────────────────

    def get_recent_applications(self, days: int = 7) -> list[Application]:
        """获取最近一段时间内投递的记录。

        Args:
            days: 最近几天（默认 7 天）。

        Returns:
            近期的 Application 列表。
        """
        cutoff = date.today() - timedelta(days=days)
        all_apps = self._store.list()
        return [app for app in all_apps if app.submit_date >= cutoff]

    def get_applications_by_status(self, status: ApplicationStatus) -> list[Application]:
        """按状态获取投递记录。

        Args:
            status: 目标状态。

        Returns:
            对应状态的 Application 列表。
        """
        return self._store.list(filter_status=status.value)


# 模块级单例
tracking_service = TrackingService()
