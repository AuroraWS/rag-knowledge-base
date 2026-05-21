"""面试准备服务 — 匹配分析 / 复习计划 / 每日回顾。

使用 RAGPipeline 进行简历-JD 匹配分析，并基于匹配结果生成
结构化的面试准备计划和每日复习内容。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from app.models.schema import MatchResult
from app.rag.generator import LLMGenerator
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── 系统提示词 ──────────────────────────────────────

PREP_PLAN_SYSTEM_PROMPT = """你是一位专业的面试准备教练。请根据简历-JD匹配分析结果和面试日期，
为用户生成一份天级的面试准备计划。

输出必须是一个 JSON 对象（不要包含 markdown 代码块标记），包含以下字段：

{
    "overall_assessment": "对用户整体匹配状况的简要评估",
    "focus_areas": ["需要重点准备的领域1", "领域2"],
    "daily_plans": [
        {
            "day": 1,
            "date": "YYYY-MM-DD",
            "title": "当日主题",
            "tasks": ["任务1", "任务2"],
            "estimated_minutes": 60
        }
    ]
}

注意：
- daily_plans 天数从今天到面试前一天
- 每天的任务可量化、可执行
- 优先安排薄弱环节的复习
"""

DAILY_REVIEW_SYSTEM_PROMPT = """你是一位面试准备教练。请根据用户的面试准备计划，生成今天的复习内容。

输出格式为 Markdown 文本，包含：
1. 今日复习主题
2. 重点回顾内容（基于匹配分析中的薄弱环节）
3. 练习题目或思考题（2-3个）
4. 今日小贴士
"""


class PreparationService:
    """面试准备服务 — 简历-JD 匹配、准备计划、每日回顾。"""

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        generator: Optional[LLMGenerator] = None,
    ) -> None:
        self._pipeline = pipeline or RAGPipeline()
        self._generator = generator or LLMGenerator()

    # ── 匹配分析 ────────────────────────────────────

    async def analyze_match(
        self,
        resume_data: dict[str, Any],
        jd_data: dict[str, Any],
    ) -> MatchResult:
        """分析简历与 JD 的匹配度。

        Args:
            resume_data: 简历数据（含 raw_text 或结构化字段）。
            jd_data: JD 数据（含 raw_text 或结构化字段）。

        Returns:
            MatchResult 结构化匹配结果。
        """
        # 提取纯文本
        resume_text = self._extract_text(resume_data)
        jd_text = self._extract_text(jd_data)

        # 获取结构化模型（如果有）
        resume_model = resume_data.get("_model") if isinstance(resume_data, dict) else None
        jd_model = jd_data.get("_model") if isinstance(jd_data, dict) else None

        try:
            result = await self._pipeline.match_resume_jd_structured(
                resume_text=resume_text,
                jd_text=jd_text,
                resume_model=resume_model,
                jd_model=jd_model,
            )
            return result

        except Exception as e:
            logger.error("匹配分析失败: %s", e, exc_info=True)
            # 降级返回
            return MatchResult(
                resume_name="未知",
                jd_title=jd_data.get("title", "未知") if isinstance(jd_data, dict) else "未知",
                company=jd_data.get("company", "未知") if isinstance(jd_data, dict) else "未知",
                match_score=0.0,
                matched_skills=[],
                missing_skills=[],
                analysis=f"匹配分析生成失败: {e}",
            )

    # ── 准备计划生成 ────────────────────────────────

    async def generate_prep_plan(
        self,
        interview_date: date,
        match_result: MatchResult,
    ) -> dict[str, Any]:
        """根据面试日期和匹配结果生成天级准备计划。

        Args:
            interview_date: 面试日期。
            match_result: 匹配分析结果。

        Returns:
            包含 overall_assessment, focus_areas, daily_plans 的字典。
        """
        today = date.today()
        days_until_interview = (interview_date - today).days

        if days_until_interview <= 0:
            # 面试在今天或已过
            return {
                "overall_assessment": "面试时间已到或已过，请查看每日回顾进行最后冲刺。",
                "focus_areas": [match_result.analysis[:100] if match_result.analysis else "准备面试"],
                "daily_plans": [
                    {
                        "day": 0,
                        "date": today.isoformat(),
                        "title": "面试当天准备",
                        "tasks": [
                            "复习关键技能和项目经验",
                            "准备面试中可能遇到的问题",
                            "提前测试设备（如果是线上面试）",
                            "深呼吸，保持自信",
                        ],
                        "estimated_minutes": 90,
                    }
                ],
            }

        # 构建匹配摘要
        match_summary = (
            f"匹配分数: {match_result.match_score:.2f}\n"
            f"匹配技能: {', '.join(match_result.matched_skills)}\n"
            f"缺失技能: {', '.join(match_result.missing_skills)}\n"
            f"详细分析: {match_result.analysis}"
        )

        prompt = f"""请根据以下信息生成面试准备计划。

===== 匹配分析结果 =====
{match_summary}

===== 面试日期 =====
{interview_date.isoformat()}

===== 今天日期 =====
{today.isoformat()}

===== 剩余天数 =====
{days_until_interview} 天

请生成从今天到面试日期的逐日准备计划。"""

        try:
            result = await self._generator.generate_structured(
                prompt=prompt,
                system_prompt=PREP_PLAN_SYSTEM_PROMPT,
                temperature=0.7,
            )

            if isinstance(result, str):
                result = json.loads(result)

            return {
                "overall_assessment": result.get("overall_assessment", ""),
                "focus_areas": result.get("focus_areas", []),
                "daily_plans": result.get("daily_plans", []),
            }

        except Exception as e:
            logger.error("准备计划生成失败: %s", e, exc_info=True)
            return {
                "overall_assessment": "请自行安排复习计划",
                "focus_areas": match_result.missing_skills or ["综合准备"],
                "daily_plans": [],
            }

    # ── 每日回顾 ────────────────────────────────────

    async def generate_daily_review(
        self,
        plan: dict[str, Any],
        day_offset: int,
    ) -> str:
        """根据准备计划和当前进度生成今日复习内容。

        Args:
            plan: generate_prep_plan 返回的计划字典。
            day_offset: 从计划开始的第几天（0=第一天）。

        Returns:
            Markdown 格式的今日复习内容。
        """
        daily_plans = plan.get("daily_plans", [])

        # 找到当天的计划
        today_plan = None
        for dp in daily_plans:
            if dp.get("day") == day_offset:
                today_plan = dp
                break

        if today_plan is None:
            return self._generate_default_review(plan, day_offset)

        focus_areas = plan.get("focus_areas", [])
        overall = plan.get("overall_assessment", "")

        prompt = f"""请为今日的面试准备生成复习内容。

===== 整体评估 =====
{overall}

===== 今日计划 =====
日期: {today_plan.get('date', '')}
主题: {today_plan.get('title', '')}
任务: {', '.join(today_plan.get('tasks', []))}

===== 重点领域 =====
{', '.join(focus_areas)}

请生成详细的今日复习内容。"""

        try:
            result = await self._generator.generate(
                prompt=prompt,
                system_prompt=DAILY_REVIEW_SYSTEM_PROMPT,
                temperature=0.7,
            )
            return result

        except Exception as e:
            logger.error("每日回顾生成失败: %s", e, exc_info=True)
            return self._generate_default_review(plan, day_offset)

    def _generate_default_review(self, plan: dict[str, Any], day_offset: int) -> str:
        """生成默认的复习内容（当 LLM 调用失败时）。"""
        focus = plan.get("focus_areas", ["综合复习"])
        tasks = []

        for area in focus:
            tasks.append(f"- 复习 {area}")
            tasks.append(f"  - 回顾相关技能知识点")
            tasks.append(f"  - 思考面试可能被问到的问题")
            tasks.append(f"  - 准备对应的STAR案例")

        return f"""# 📚 面试准备 · 第 {day_offset + 1} 天

## 今日复习主题
重点复习以下领域：{', '.join(focus)}

## 复习内容
{chr(10).join(tasks)}

## 练习题目
1. 请用 STAR 格式介绍你的核心项目
2. 你对这个岗位理解是什么？
3. 你为什么适合这个岗位？

## 今日小贴士
💡 每次复习后，尝试用自己的话复述一遍，加深记忆。
"""

    # ── 辅助方法 ────────────────────────────────────

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """从数据结构中提取纯文本。"""
        # 优先使用 raw_text
        if isinstance(data, dict):
            raw = data.get("raw_text") or data.get("raw_text", "")
            if raw:
                return raw

            # 尝试从结构化字段拼装
            parts = []
            if "company" in data:
                parts.append(f"公司: {data.get('company', '')}")
            if "title" in data:
                parts.append(f"岗位: {data.get('title', '')}")
            if "responsibilities" in data:
                parts.append(f"职责: {'; '.join(data['responsibilities'])}")
            if "requirements" in data:
                reqs = data["requirements"]
                if reqs and isinstance(reqs[0], dict):
                    parts.append(f"要求: {'; '.join(r.get('content', '') for r in reqs)}")
            if "name" in data:
                parts.append(f"姓名: {data.get('name', '')}")
            if "skills" in data:
                parts.append(f"技能: {', '.join(data.get('skills', []))}")

            return "\n".join(parts) if parts else json.dumps(data, ensure_ascii=False)

        return str(data)

    async def close(self) -> None:
        """释放资源。"""
        await self._pipeline.close()
        await self._generator.close()


# 模块级单例
preparation_service = PreparationService()
