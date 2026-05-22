"""投递记录存储 — SQLite 持久化。

管理用户的招聘投递记录，支持 CRUD、状态跟踪、定时提醒等功能。
自动管理表结构，首次使用时创建表。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schema import Application, ApplicationStatus, StatusChange


# ── SQL 常量 ─────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS applications (
    id          TEXT PRIMARY KEY,
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    jd_text     TEXT DEFAULT '',
    channel     TEXT NOT NULL DEFAULT '',
    resume_version TEXT DEFAULT '',
    cover_letter   TEXT DEFAULT '',
    submit_date TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT '已投递待反馈',
    last_check  TEXT DEFAULT NULL,
    next_check  TEXT DEFAULT NULL,
    timeline    TEXT DEFAULT '[]',
    notes       TEXT DEFAULT ''
)
"""

_INSERT_SQL = """
INSERT INTO applications
    (id, company, title, location, url, jd_text, channel,
     resume_version, cover_letter, submit_date, status,
     last_check, next_check, timeline, notes)
VALUES
    (?, ?, ?, ?, ?, ?, ?,
     ?, ?, ?, ?,
     ?, ?, ?, ?)
"""

_UPDATE_SQL = """
UPDATE applications SET
    company=?, title=?, location=?, url=?, jd_text=?, channel=?,
    resume_version=?, cover_letter=?, submit_date=?, status=?,
    last_check=?, next_check=?, timeline=?, notes=?
WHERE id=?
"""

_SELECT_BY_ID_SQL = "SELECT * FROM applications WHERE id=?"
_SELECT_ALL_SQL = "SELECT * FROM applications ORDER BY submit_date DESC"
_SELECT_BY_STATUS_SQL = (
    "SELECT * FROM applications WHERE status=? ORDER BY submit_date DESC"
)
_DELETE_SQL = "DELETE FROM applications WHERE id=?"


class ApplicationStore:
    """投递记录存储，基于 SQLite。

    自动管理表结构的创建和迁移。所有日期以 ISO 格式（``YYYY-MM-DD``）
    字符串存储，timeline 以 JSON 数组存储。
    """

    def __init__(self) -> None:
        self._db_path: str = settings.sqlite_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── 连接管理 ───────────────────────────────────

    def _ensure_dir(self) -> None:
        """确保 SQLite 文件所在目录存在。"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（懒初始化）。"""
        if self._conn is None:
            self._ensure_dir()
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_table()
        return self._conn

    def _init_table(self) -> None:
        """确保表存在。"""
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── 序列化 / 反序列化 ──────────────────────────

    @staticmethod
    def _row_to_application(row: sqlite3.Row) -> Application:
        """将 SQLite 行转为 Application 模型。"""
        d = dict(row)
        # timeline：JSON 字符串 -> list[StatusChange]
        raw_timeline = d.pop("timeline", "[]")
        if isinstance(raw_timeline, str):
            timeline_data = json.loads(raw_timeline) if raw_timeline else []
        else:
            timeline_data = raw_timeline or []

        timeline = []
        for item in timeline_data:
            if isinstance(item, dict):
                if "status" in item and "date" in item:
                    item["status"] = ApplicationStatus(item["status"])
                    if isinstance(item["date"], str):
                        item["date"] = date.fromisoformat(item["date"])
                    # StatusChange uses alias date->change_date with populate_by_name
                    timeline.append(StatusChange(**item))

        # 日期字段
        for field in ("submit_date", "last_check", "next_check"):
            val = d.get(field)
            if val and isinstance(val, str):
                d[field] = date.fromisoformat(val)

        # 状态枚举
        if "status" in d and isinstance(d["status"], str):
            d["status"] = ApplicationStatus(d["status"])

        d["timeline"] = timeline
        return Application(**d)

    @staticmethod
    def _application_to_row(app: Application) -> dict:
        """将 Application 模型转为可写入 SQLite 的 dict。"""
        d = app.model_dump()
        # timeline：list[StatusChange] -> JSON 字符串
        if "timeline" in d and d["timeline"]:
            d["timeline"] = json.dumps(
                [
                    {
                        "status": sc["status"].value if hasattr(sc["status"], "value") else sc["status"],
                        "date": sc["change_date"].isoformat() if hasattr(sc["change_date"], "isoformat") else sc["change_date"],
                        "note": sc.get("note"),
                    }
                    for sc in d["timeline"]
                ],
                ensure_ascii=False,
            )
        else:
            d["timeline"] = "[]"
        # 日期字段转为 ISO 字符串
        for field in ("submit_date", "last_check", "next_check"):
            val = d.get(field)
            if val and hasattr(val, "isoformat"):
                d[field] = val.isoformat()
        # 状态枚举 -> 字符串
        if "status" in d and hasattr(d["status"], "value"):
            d["status"] = d["status"].value
        return d

    # ── CRUD ────────────────────────────────────────

    def add(self, app: Application) -> Application:
        """添加一条投递记录。"""
        conn = self._get_conn()
        row = self._application_to_row(app)
        conn.execute(
            _INSERT_SQL,
            (
                row["id"], row["company"], row["title"], row["location"],
                row["url"], row["jd_text"], row["channel"],
                row["resume_version"], row["cover_letter"],
                row["submit_date"], row["status"],
                row["last_check"], row["next_check"],
                row["timeline"], row["notes"],
            ),
        )
        conn.commit()
        return app

    def get(self, id: str) -> Optional[Application]:
        """根据 ID 获取一条投递记录，不存在返回 None。"""
        conn = self._get_conn()
        cursor = conn.execute(_SELECT_BY_ID_SQL, (id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_application(row)

    def update(self, app: Application) -> Application:
        """更新一条投递记录（全量覆盖）。"""
        conn = self._get_conn()
        row = self._application_to_row(app)
        conn.execute(
            _UPDATE_SQL,
            (
                row["company"], row["title"], row["location"],
                row["url"], row["jd_text"], row["channel"],
                row["resume_version"], row["cover_letter"],
                row["submit_date"], row["status"],
                row["last_check"], row["next_check"],
                row["timeline"], row["notes"],
                row["id"],
            ),
        )
        conn.commit()
        return app

    def update_status(
        self, id: str, status: ApplicationStatus, note: str = ""
    ) -> Application:
        """更新投递状态，自动追加时间线记录。

        Args:
            id: 投递记录 ID
            status: 新状态
            note: 状态变更备注

        Returns:
            更新后的 Application 对象

        Raises:
            ValueError: 指定 ID 的记录不存在
        """
        app = self.get(id)
        if app is None:
            raise ValueError(f"投递记录不存在: {id}")

        # 追加状态变更记录
        change = StatusChange(status=status, date=date.today(), note=note or None)
        app.timeline.append(change)
        app.status = status

        return self.update(app)

    def list(self, filter_status: Optional[str] = None) -> list[Application]:
        """获取投递记录列表。

        Args:
            filter_status: 可选的过滤状态（中文值，如 '面试中'），
                          为 None 时返回全部。

        Returns:
            Application 列表，按投递日期倒序。
        """
        conn = self._get_conn()
        if filter_status:
            cursor = conn.execute(_SELECT_BY_STATUS_SQL, (filter_status,))
        else:
            cursor = conn.execute(_SELECT_ALL_SQL)
        return [self._row_to_application(row) for row in cursor.fetchall()]

    def stats(self) -> dict:
        """获取投递统计信息。

        Returns:
            {
                "by_status": {"待投递": 2, "面试中": 1, ...},
                "total": 10,
                "pending_check": 3,
            }
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM applications GROUP BY status"
        )
        by_status: dict[str, int] = {}
        for row in cursor.fetchall():
            by_status[row["status"]] = row["cnt"]

        cursor2 = conn.execute(
            "SELECT COUNT(*) as cnt FROM applications WHERE next_check IS NOT NULL AND next_check <= ?",
            (date.today().isoformat(),),
        )
        pending_check = cursor2.fetchone()["cnt"]

        total = sum(by_status.values())

        return {
            "by_status": by_status,
            "total": total,
            "pending_check": pending_check,
        }

    def get_pending_check(self, days: int = 5) -> list[Application]:
        """获取需要跟进检查的投递记录。

        筛选条件：``next_check`` 不为空且 ``<= today + days``。

        Args:
            days: 提前天数，默认 5 天内的都需要检查。

        Returns:
            需要跟进的 Application 列表。
        """
        conn = self._get_conn()
        cutoff = (date.today() + timedelta(days=days)).isoformat()
        cursor = conn.execute(
            "SELECT * FROM applications WHERE next_check IS NOT NULL AND next_check <= ? ORDER BY next_check ASC",
            (cutoff,),
        )
        return [self._row_to_application(row) for row in cursor.fetchall()]

    def delete(self, id: str) -> None:
        """删除一条投递记录。"""
        conn = self._get_conn()
        conn.execute(_DELETE_SQL, (id,))
        conn.commit()


# 模块级单例
application_store = ApplicationStore()
