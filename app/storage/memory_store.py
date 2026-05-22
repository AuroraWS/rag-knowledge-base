"""字段记忆存储 — JSON 文件持久化。

系统自动记住用户在各个表单中填写的字段值，
下次遇到相同字段时自动填充。数据以 JSON 格式保存在
``settings.memory_dir`` 下，按字段键索引。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schema import FieldMemory


class MemoryStore:
    """字段记忆持久化存储。

    所有 FieldMemory 按 ``field_key`` 索引保存为 JSON 文件。
    使用前确保记忆目录存在，首次调用时自动创建。
    """

    def __init__(self) -> None:
        self._file_path = os.path.join(settings.memory_dir, "field_memories.json")
        self._memories: dict[str, FieldMemory] = {}
        self._loaded = False

    # ── 内部 I/O ──────────────────────────────────

    def _ensure_dir(self) -> None:
        """确保记忆数据目录存在。"""
        Path(settings.memory_dir).mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, FieldMemory]:
        """从 JSON 文件加载所有记忆，若文件不存在则返回空字典。"""
        if self._loaded:
            return self._memories

        self._ensure_dir()

        if os.path.isfile(self._file_path):
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._memories = {}
            for key, item in raw.items():
                # 日期时间字段反序列化
                if "first_seen" in item and isinstance(item["first_seen"], str):
                    item["first_seen"] = datetime.fromisoformat(item["first_seen"])
                if "last_updated" in item and isinstance(item["last_updated"], str):
                    item["last_updated"] = datetime.fromisoformat(item["last_updated"])
                self._memories[key] = FieldMemory(**item)
        else:
            self._memories = {}

        self._loaded = True
        return self._memories

    def _save(self) -> None:
        """将当前所有记忆写入 JSON 文件。"""
        self._ensure_dir()
        raw = {
            key: mem.model_dump(mode="json")
            for key, mem in self._memories.items()
        }
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    # ── 读写操作 ──────────────────────────────────

    def get(self, key: str) -> Optional[FieldMemory]:
        """根据字段键获取记忆。"""
        memories = self._load()
        return memories.get(key)

    def get_value(self, key: str) -> Optional[str]:
        """快捷方法：直接获取字段记忆的值。

        Args:
            key: 字段键

        Returns:
            字段值字符串，若不存在返回 None。
        """
        memory = self.get(key)
        return memory.value if memory else None

    def set(self, memory: FieldMemory) -> FieldMemory:
        """新增或更新一条记忆（upsert）。

        如果 ``field_key`` 已存在，更新其值并刷新 ``last_updated``；
        否则直接插入。

        Args:
            memory: 要保存的 FieldMemory 对象

        Returns:
            保存后的 FieldMemory 对象。
        """
        memories = self._load()
        existing = memories.get(memory.field_key)

        if existing:
            # 更新现有记录（保留 first_seen）
            existing.value = memory.value
            existing.last_updated = datetime.now()
            if memory.source_context:
                existing.source_context = memory.source_context
            if memory.confidence is not None:
                existing.confidence = max(existing.confidence, memory.confidence)
            if memory.field_label:
                existing.field_label = memory.field_label
            result = existing
        else:
            # 新记录
            now = datetime.now()
            memory.first_seen = now
            memory.last_updated = now
            memories[memory.field_key] = memory
            result = memory

        self._save()
        return result

    def remove(self, key: str) -> bool:
        """删除一条记忆。

        Args:
            key: 待删除的字段键

        Returns:
            True 表示存在且已删除，False 表示键不存在。
        """
        memories = self._load()
        if key in memories:
            del memories[key]
            self._save()
            return True
        return False

    def find_by_label(self, label: str) -> list[FieldMemory]:
        """按字段标签（关键词子串匹配，不区分大小写）查找记忆。

        Args:
            label: 搜索关键词

        Returns:
            匹配的记忆列表。
        """
        memories = self._load()
        lower_label = label.lower()
        return [
            m for m in memories.values()
            if lower_label in m.field_label.lower()
        ]

    def all(self) -> list[FieldMemory]:
        """返回所有记忆，按 ``last_updated`` 倒序排列。"""
        memories = self._load()
        return sorted(
            memories.values(),
            key=lambda m: m.last_updated,
            reverse=True,
        )


# 模块级单例
memory_store = MemoryStore()
