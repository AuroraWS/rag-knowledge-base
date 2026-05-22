"""APScheduler 定时任务调度器 — 每日回顾 / 每日日志。

在后台运行定时任务：
- 09:30: 推送今日待办和每日复习
- 22:00: 生成当日投递日志

WeChat bot 推送为占位实现（打印到日志），后续可替换为真实微信/飞书 bot。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

from app.config import settings
from app.services.preparation_service import preparation_service
from app.services.tracking_service import tracking_service

logger = logging.getLogger(__name__)

# ── 检查 APScheduler 是否可用 ──────────────────────

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    logger.warning(
        "APScheduler 未安装。定时任务将不可用。"
        " 安装: pip install apscheduler"
    )


class JobScheduler:
    """定时任务调度器 — 管理每日回顾推送和日志生成。

    使用 APScheduler AsyncIOScheduler 在异步事件循环中运行。
    """

    def __init__(self) -> None:
        self._scheduler: Any = None
        self._running = False

    # ── 启动 / 停止 ───────────────────────────────

    def start_scheduler(self) -> None:
        """启动后台调度器。"""
        if not _APSCHEDULER_AVAILABLE:
            logger.warning("APScheduler 未安装，无法启动调度器")
            return

        if self._running:
            logger.info("调度器已在运行中")
            return

        self._scheduler = AsyncIOScheduler()
        self._running = True

        # 注册定时任务
        self._register_jobs()

        self._scheduler.start()
        logger.info(
            "定时调度器已启动 | 每日回顾: %s | 每日日志: %s",
            settings.schedule_review_time,
            settings.daily_log_time,
        )

    def _register_jobs(self) -> None:
        """注册所有定时任务。"""
        if not self._scheduler:
            return

        # 每日回顾推送（09:30）
        review_hour, review_min = self._parse_time(settings.schedule_review_time)
        self._scheduler.add_job(
            self._run_daily_review_job,
            CronTrigger(hour=review_hour, minute=review_min),
            id="daily_review",
            name="每日回顾推送",
            coalesce=True,
            max_instances=1,
        )

        # 每日日志生成（22:00）
        log_hour, log_min = self._parse_time(settings.daily_log_time)
        self._scheduler.add_job(
            self._run_daily_log_job,
            CronTrigger(hour=log_hour, minute=log_min),
            id="daily_log",
            name="每日日志生成",
            coalesce=True,
            max_instances=1,
        )

        logger.info("已注册 %d 个定时任务", len(self._scheduler.get_jobs()))

    # ── 定时任务回调 ──────────────────────────────

    async def _run_daily_review_job(self) -> None:
        """每日回顾推送任务（09:30 执行）。"""
        logger.info("===== 开始执行每日回顾任务 =====")

        try:
            # Step 1: 获取今日待办
            todo_items = tracking_service.get_today_todo()

            # Step 2: 获取状态摘要
            summary = tracking_service.get_status_summary()

            # Step 3: 推送摘要
            message = self._build_review_message(todo_items, summary)

            # Step 4: 通过 WeChat bot 推送（占位）
            await self._send_wechat_message(message)

            logger.info("每日回顾推送完成 | 待办: %d 项", len(todo_items))

        except Exception as e:
            logger.error("每日回顾任务执行失败: %s", e, exc_info=True)

    async def _run_daily_log_job(self) -> None:
        """每日日志生成任务（22:00 执行）。"""
        logger.info("===== 开始执行每日日志生成任务 =====")

        try:
            today = date.today()
            today_str = today.isoformat()

            # Step 1: 获取今日投递记录
            recent = tracking_service.get_recent_applications(days=1)

            # Step 2: 获取状态分布
            summary = tracking_service.get_status_summary()

            # Step 3: 构建日志内容
            log_content = self._build_log_content(today_str, recent, summary)

            # Step 4: 推送日志
            await self._send_wechat_message(f"📝 每日投递日志 ({today_str})\n\n{log_content}")

            # Step 5: 写入本地日志文件
            self._write_log_file(today_str, log_content)

            logger.info("每日日志生成完成 | 今日投递: %d 条", len(recent))

        except Exception as e:
            logger.error("每日日志生成任务执行失败: %s", e, exc_info=True)

    # ── 消息构建 ──────────────────────────────────

    @staticmethod
    def _build_review_message(
        todo_items: list[dict],
        summary: dict,
    ) -> str:
        """构建每日回顾推送消息。"""
        lines = ["☀️ 早安！今日待办清单", "=" * 20]

        # 状态摘要
        lines.append(f"\n📊 投递概况")
        lines.append(f"  总投递: {summary.get('total', 0)}")
        lines.append(f"  面试中: {summary.get('interview_count', 0)}")
        lines.append(f"  待检查: {summary.get('pending_check_count', 0)}")

        if todo_items:
            lines.append(f"\n📋 今日待办 ({len(todo_items)} 项)")

            for item in todo_items:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                icon = priority_icon.get(item.get("priority", "low"), "⚪")
                lines.append(
                    f"  {icon} [{item.get('type', 'task')}] "
                    f"{item.get('company', '')} - {item.get('title', '')}"
                )
        else:
            lines.append("\n✅ 今日没有待办事项")

        return "\n".join(lines)

    @staticmethod
    def _build_log_content(
        today_str: str,
        recent: list,
        summary: dict,
    ) -> str:
        """构建每日日志内容。"""
        lines = [f"日期: {today_str}"]

        if recent:
            lines.append(f"\n今日投递 ({len(recent)} 条):")
            for app in recent:
                lines.append(f"  • {app.company} - {app.title} ({app.status.value})")
        else:
            lines.append("\n今日无新增投递")

        lines.append(f"\n状态分布:")
        for status, count in summary.get("by_status", {}).items():
            lines.append(f"  {status}: {count}")

        return "\n".join(lines)

    @staticmethod
    def _write_log_file(date_str: str, content: str) -> None:
        """将日志写入本地文件。"""
        import os
        from pathlib import Path

        log_dir = os.path.join(settings.log_dir, "daily")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_path = os.path.join(log_dir, f"{date_str}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("日志已写入: %s", file_path)

    # ── WeChat Bot（占位） ────────────────────────

    async def _send_wechat_message(self, message: str) -> None:
        """通过 WeChat bot 发送消息。"""
        from app.gateway.push import push_manager

        await push_manager.send("default_user", message)

    # ── 任务管理 ──────────────────────────────────

    def add_daily_review_job(self, time: str = "09:30") -> None:
        """添加（或更新）每日回顾推送任务。

        Args:
            time: 执行时间，格式 "HH:MM"。
        """
        if not self._scheduler:
            logger.warning("调度器未启动")
            return

        hour, minute = self._parse_time(time)

        # 删除旧任务
        self._scheduler.remove_job("daily_review")

        self._scheduler.add_job(
            self._run_daily_review_job,
            CronTrigger(hour=hour, minute=minute),
            id="daily_review",
            name="每日回顾推送",
            coalesce=True,
            max_instances=1,
        )
        logger.info("每日回顾任务已更新: %s", time)

    def add_daily_log_job(self, time: str = "22:00") -> None:
        """添加（或更新）每日日志生成任务。

        Args:
            time: 执行时间，格式 "HH:MM"。
        """
        if not self._scheduler:
            logger.warning("调度器未启动")
            return

        hour, minute = self._parse_time(time)

        # 删除旧任务
        self._scheduler.remove_job("daily_log")

        self._scheduler.add_job(
            self._run_daily_log_job,
            CronTrigger(hour=hour, minute=minute),
            id="daily_log",
            name="每日日志生成",
            coalesce=True,
            max_instances=1,
        )
        logger.info("每日日志任务已更新: %s", time)

    # ── 停止 ──────────────────────────────────────

    def stop_scheduler(self) -> None:
        """停止调度器。"""
        if not self._scheduler:
            return

        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("定时调度器已停止")

    # ── 工具方法 ──────────────────────────────────

    @staticmethod
    def _parse_time(time_str: str) -> tuple[int, int]:
        """解析 "HH:MM" 格式的时间字符串。

        Returns:
            (hour, minute) 整数元组。
        """
        try:
            parts = time_str.strip().split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return hour, minute
        except (ValueError, IndexError):
            logger.warning("时间格式错误: %s，使用默认值 09:30", time_str)
            return 9, 30

    # ── 上下文管理器 ──────────────────────────────

    async def __aenter__(self) -> JobScheduler:
        self.start_scheduler()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.stop_scheduler()


# 模块级单例
job_scheduler = JobScheduler()
