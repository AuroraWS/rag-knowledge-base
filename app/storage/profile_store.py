"""个人简历信息存储 — JSON 文件持久化。

存储以下简历模块：
- 个人基本信息 (PersonalInfo)
- 教育经历 (Education)
- 工作经历 (WorkExperience)
- 项目经历 (Project)
- 证书/资质 (Certificate)

首次使用若无数据文件，自动以王爽的示例数据初始化。
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schema import (
    Certificate,
    Education,
    PersonalInfo,
    Project,
    WorkExperience,
)

# ──────────────────────────────────────────────
# 示例默认数据（王爽）
# ──────────────────────────────────────────────

_DEFAULT_PERSONAL_INFO = PersonalInfo(
    name="王爽",
    phone="13032113120",
    email="aurora001@foxmail.com",
    wechat="qiuqiuzi007",
    target_location=["上海", "北京"],
    target_salary="15k-20k",
    earliest_start_date="随时",
)

_DEFAULT_EDUCATION: list[Education] = [
    Education(
        school="伯明翰大学",
        degree="硕士",
        major="计算机科学",
        start_date="2023.09",
        end_date="2025.01",
        is_overseas=True,
    ),
    Education(
        school="同济大学",
        degree="学士",
        major="电子信息工程",
        start_date="2015.09",
        end_date="2020.07",
        is_overseas=False,
    ),
]

_DEFAULT_WORK_EXPERIENCE: list[WorkExperience] = [
    WorkExperience(
        company="招商证券金融科技中心",
        title="算法开发工程师（实习）",
        department=None,
        start_date="2023.02",
        end_date="2023.08",
        is_current=False,
        responsibilities=[
            "设计并部署金融客服RAG系统",
            "对多个LLM进行多维度评测",
            "主导数据清洗去重与结构化处理",
        ],
        achievements=[
            "BM25+Sentence-BERT双引擎检索，FAISS索引10万级语料毫秒级召回，错误响应率降低50%",
            "覆盖GPT-3.5、LLaMA等5个模型、10项NLP任务",
            "TF-IDF去重降低冗余率40%",
        ],
        tech_stack=["BM25", "Sentence-BERT", "FAISS", "GPT-3.5", "LLaMA"],
    ),
]

_DEFAULT_PROJECTS: list[Project] = [
    Project(
        name="RAG知识库系统 + 多Agent工作流",
        role="独立开发者",
        start_date="2025.09",
        end_date=None,
        description="基于LangChain的多Agent RAG系统，集成FAISS向量检索与DeepSeek API",
        tech_stack=[
            "LangChain",
            "bge-small-zh-v1.5",
            "bge-reranker-v2-m3",
            "FAISS",
            "DeepSeek API",
            "FastAPI",
            "Docker",
        ],
        highlights=[
            "语义分块+递归分块策略，FAISS HNSW-PQ混合索引，P99检索延迟<200ms",
            "基于LangChain设计多Agent编排：查询改写、多步检索规划、答案合成带护栏",
            "集成Claude Code + 结构化skills/playbooks到开发流程",
            "搭建评估管线：日志记录、链路追踪、A/B测试框架",
            "FastAPI异步服务+Docker部署，RESTful API支持并发",
        ],
    ),
    Project(
        name="人体动作识别模型优化",
        role="算法研究员",
        start_date="2024.01",
        end_date="2024.05",
        description="人体骨架动作识别模型优化与边缘端部署",
        tech_stack=["PyTorch", "ResNet-50", "Bi-LSTM", "NVIDIA Jetson TX2"],
        highlights=[
            "ResNet-50（空间特征）+ Bi-LSTM（时序建模），NTU RGB+D 120数据集92.5%准确率",
            "时间扭曲/帧插值增强方案，速度扰动场景性能衰减从15%降至5%",
            "PyTorch Profiler优化推理管线，延迟降低30%，Jetson TX2上25 FPS实时处理",
        ],
    ),
    Project(
        name="Elasticsearch新闻搜索引擎",
        role="独立开发者",
        start_date="2020.09",
        end_date=None,
        description="基于Elasticsearch 8.x的全栈新闻搜索系统",
        tech_stack=["Elasticsearch", "BM25", "Python"],
        highlights=[
            "搜索准确率提升20%，查询速度较传统SQL提升30%",
        ],
    ),
]

_DEFAULT_CERTIFICATES: list[Certificate] = []


class ProfileStore:
    """用户简历信息存储。

    数据以 JSON 格式保存在 ``settings.profile_dir`` 下。
    使用前确保目录存在，会在首次调用时自动创建。
    """

    def __init__(self) -> None:
        self._file_path = os.path.join(settings.profile_dir, "profile.json")
        self._data: dict = {}
        self._loaded = False

    # ── 内部 I/O ──────────────────────────────────────

    def _ensure_dir(self) -> None:
        """确保数据目录存在。"""
        Path(settings.profile_dir).mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        """从 JSON 文件加载全部数据，若文件不存在则创建默认数据。"""
        if self._loaded:
            return self._data

        self._ensure_dir()

        if not os.path.isfile(self._file_path):
            self._data = self._build_default_data()
            self._save()
        else:
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._data = self._decode(raw)

        self._loaded = True
        return self._data

    def _save(self) -> None:
        """将当前数据写入 JSON 文件。"""
        self._ensure_dir()
        raw = self._encode(self._data)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _build_default_data() -> dict:
        """构建包含王爽示例数据的初始字典（内部格式）。"""
        return {
            "personal_info": _DEFAULT_PERSONAL_INFO.model_dump(),
            "education": [e.model_dump() for e in _DEFAULT_EDUCATION],
            "work_experience": [w.model_dump() for w in _DEFAULT_WORK_EXPERIENCE],
            "projects": [p.model_dump() for p in _DEFAULT_PROJECTS],
            "certificates": [c.model_dump() for c in _DEFAULT_CERTIFICATES],
        }

    @staticmethod
    def _encode(data: dict) -> dict:
        """将内部 dict（含 Pydantic 对象）编码为可 JSON 序列化的 dict。"""
        # 如果已全部是 dict，直接返回
        return data

    @staticmethod
    def _decode(raw: dict) -> dict:
        """将 JSON 反序列化后的 dict 转为内部 dict（不变，后续按需构造模型）。"""
        # 数据在 getter 中按需构造模型对象
        return raw

    # ── 个人基本信息 ───────────────────────────────────

    def get_personal_info(self) -> Optional[PersonalInfo]:
        """获取个人基本信息。"""
        data = self._load()
        raw = data.get("personal_info")
        if not raw:
            return None
        return self._decode_personal_info(raw)

    def update_personal_info(self, info: PersonalInfo) -> PersonalInfo:
        """更新个人基本信息，返回更新后的对象。"""
        data = self._load()
        data["personal_info"] = info.model_dump()
        self._save()
        return info

    @staticmethod
    def _decode_personal_info(raw: dict) -> PersonalInfo:
        """将 dict 转为 PersonalInfo，处理日期字段。"""
        if "birthday" in raw and raw["birthday"] and isinstance(raw["birthday"], str):
            raw["birthday"] = date.fromisoformat(raw["birthday"])
        return PersonalInfo(**raw)

    # ── 教育经历 ───────────────────────────────────────

    def get_education(self) -> list[Education]:
        """获取所有教育经历。"""
        data = self._load()
        return [Education(**e) for e in data.get("education", [])]

    def add_education(self, edu: Education) -> Education:
        """添加一条教育经历。"""
        data = self._load()
        data.setdefault("education", []).append(edu.model_dump())
        self._save()
        return edu

    def update_education(self, index: int, edu: Education) -> Education:
        """更新指定索引的教育经历。"""
        data = self._load()
        edu_list = data.get("education", [])
        if index < 0 or index >= len(edu_list):
            raise IndexError(
                f"教育经历索引 {index} 超出范围（共 {len(edu_list)} 条）"
            )
        edu_list[index] = edu.model_dump()
        self._save()
        return edu

    def remove_education(self, index: int) -> None:
        """删除指定索引的教育经历。"""
        data = self._load()
        edu_list = data.get("education", [])
        if index < 0 or index >= len(edu_list):
            raise IndexError(
                f"教育经历索引 {index} 超出范围（共 {len(edu_list)} 条）"
            )
        edu_list.pop(index)
        self._save()

    # ── 工作经历 ───────────────────────────────────────

    def get_work_experience(self) -> list[WorkExperience]:
        """获取所有工作经历。"""
        data = self._load()
        return [WorkExperience(**w) for w in data.get("work_experience", [])]

    def add_work_experience(self, exp: WorkExperience) -> WorkExperience:
        """添加一条工作经历。"""
        data = self._load()
        data.setdefault("work_experience", []).append(exp.model_dump())
        self._save()
        return exp

    # ── 项目经历 ───────────────────────────────────────

    def get_projects(self) -> list[Project]:
        """获取所有项目经历。"""
        data = self._load()
        return [Project(**p) for p in data.get("projects", [])]

    def add_project(self, proj: Project) -> Project:
        """添加一条项目经历。"""
        data = self._load()
        data.setdefault("projects", []).append(proj.model_dump())
        self._save()
        return proj

    # ── 证书/资质 ──────────────────────────────────────

    def get_certificates(self) -> list[Certificate]:
        """获取所有证书/资质。"""
        data = self._load()
        return [Certificate(**c) for c in data.get("certificates", [])]

    def add_certificate(self, cert: Certificate) -> Certificate:
        """添加一条证书/资质。"""
        data = self._load()
        data.setdefault("certificates", []).append(cert.model_dump())
        self._save()
        return cert

    # ── 全量读写 ───────────────────────────────────────

    def get_all(self) -> dict:
        """返回包含所有简历数据的字典（模型对象已反序列化）。"""
        return {
            "personal_info": self.get_personal_info(),
            "education": self.get_education(),
            "work_experience": self.get_work_experience(),
            "projects": self.get_projects(),
            "certificates": self.get_certificates(),
        }


# 模块级单例
profile_store = ProfileStore()
