"""LangGraph 工作流定义 — 投递跟踪 / 面试准备。

定义了两个核心工作流：
1. ApplicationTrackingWorkflow: 投递跟踪处理管线
2. InterviewPrepWorkflow: 面试准备分析管线

每个工作流使用 LangGraph StateGraph 定义状态转换和步骤编排。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from app.models.schema import Application, ApplicationStatus, MatchResult
from app.services.preparation_service import preparation_service
from app.services.tracking_service import tracking_service

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 检查 LangGraph 是否可用
# ═══════════════════════════════════════════════════════

try:
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning(
        "LangGraph 未安装。工作流将以回退模式运行（顺序执行各步骤）。"
        " 安装: pip install langgraph"
    )

# ═══════════════════════════════════════════════════════
# 共享类型
# ═══════════════════════════════════════════════════════

if _LANGGRAPH_AVAILABLE:

    class ApplicationTrackingState(TypedDict):
        """投递跟踪工作流的状态。"""

        application: Optional[dict]          # 当前处理的投递记录
        company: str                         # 公司名
        title: str                           # 岗位名
        status: str                          # 当前状态
        needs_check: bool                    # 是否需要检查
        check_result: Optional[str]          # 检查结果
        next_action: Optional[str]           # 推荐的下一个动作
        errors: list[str]                    # 错误列表

    class InterviewPrepState(TypedDict):
        """面试准备工作流的状态。"""

        resume_text: str                     # 简历文本
        jd_text: str                         # JD 文本
        interview_date: Optional[str]        # 面试日期
        match_result: Optional[dict]         # 匹配分析结果
        prep_plan: Optional[dict]            # 准备计划
        daily_review: Optional[str]          # 每日回顾
        errors: list[str]                    # 错误列表


# ═══════════════════════════════════════════════════════
# 1. 投递跟踪工作流
# ═══════════════════════════════════════════════════════

class ApplicationTrackingWorkflow:
    """投递跟踪处理工作流。

    步骤：
    1. 加载投递记录
    2. 判断是否需要跟进
    3. 生成检查建议
    4. 输出结果
    """

    def __init__(self) -> None:
        self._graph: Any = None
        self._build_graph()

    def _build_graph(self) -> None:
        """构建 LangGraph StateGraph。"""
        if not _LANGGRAPH_AVAILABLE:
            logger.info("LangGraph 不可用，工作流使用回退顺序模式")
            return

        workflow = StateGraph(ApplicationTrackingState)

        # 定义节点
        workflow.add_node("load_application", self._load_application)
        workflow.add_node("check_status", self._check_status)
        workflow.add_node("generate_suggestion", self._generate_suggestion)
        workflow.add_node("output_result", self._output_result)

        # 设置入口
        workflow.set_entry_point("load_application")

        # 添加边
        workflow.add_edge("load_application", "check_status")

        # 条件边：根据是否需要检查分支
        workflow.add_conditional_edges(
            "check_status",
            self._route_after_check,
            {
                "needs_check": "generate_suggestion",
                "no_action": "output_result",
            },
        )

        workflow.add_edge("generate_suggestion", "output_result")
        workflow.add_edge("output_result", END)

        self._graph = workflow.compile()

    # ── 节点函数 ──────────────────────────────────

    @staticmethod
    def _load_application(state: ApplicationTrackingState) -> dict:
        """加载投递记录详情。"""
        app_data = state.get("application")
        if not app_data:
            return {"errors": ["未提供投递记录"], "next_action": "无操作"}

        company = app_data.get("company", "未知公司")
        title = app_data.get("title", "未知岗位")
        status = app_data.get("status", "未知")

        logger.info("加载投递记录: %s - %s (%s)", company, title, status)

        return {
            "company": company,
            "title": title,
            "status": status,
        }

    @staticmethod
    def _check_status(state: ApplicationTrackingState) -> dict:
        """检查投递状态，判断是否需要跟进。"""
        status_str = state.get("status", "")
        app_data = state.get("application", {})

        needs_check = False
        check_result = ""

        try:
            status = ApplicationStatus(status_str)
        except ValueError:
            status = None

        if status == ApplicationStatus.APPLIED:
            needs_check = True
            check_result = "已投递待反馈，建议跟进检查"
        elif status == ApplicationStatus.INTERVIEWING:
            needs_check = True
            check_result = "面试中，请关注面试进度"
        elif status == ApplicationStatus.EXAM:
            needs_check = True
            check_result = "已收到笔试，请准备参加"
        elif status == ApplicationStatus.PENDING:
            needs_check = True
            check_result = "待投递，请尽快完成投递"
        elif status == ApplicationStatus.REJECTED:
            check_result = "已拒绝，无需跟进"
        elif status == ApplicationStatus.OFFER:
            check_result = "已拿到 Offer，进入后续流程"
        else:
            check_result = f"未知状态: {status_str}"

        return {
            "needs_check": needs_check,
            "check_result": check_result,
        }

    @staticmethod
    def _route_after_check(state: ApplicationTrackingState) -> str:
        """路由决策：是否需要生成建议。"""
        return "needs_check" if state.get("needs_check") else "no_action"

    @staticmethod
    def _generate_suggestion(state: ApplicationTrackingState) -> dict:
        """根据状态生成下一动作建议。"""
        status_str = state.get("status", "")
        app_data = state.get("application", {})

        try:
            status = ApplicationStatus(status_str)
        except ValueError:
            status = None

        suggestions = {
            ApplicationStatus.PENDING: "准备简历和求职信，尽快完成投递",
            ApplicationStatus.APPLIED: "等待 5-7 个工作日后若无回复，考虑发送跟进邮件",
            ApplicationStatus.EXAM: "复习相关技术栈，完成笔试",
            ApplicationStatus.INTERVIEWING: "复习项目经验和匹配技能，准备面试",
        }

        next_action = suggestions.get(status, "保持关注") if status else "保持关注"

        logger.info("建议动作 - %s: %s", state.get("company"), next_action)

        return {"next_action": next_action}

    @staticmethod
    def _output_result(state: ApplicationTrackingState) -> dict:
        """输出最终结果。"""
        logger.info(
            "工作流完成: %s - %s | 需要检查: %s | 建议: %s",
            state.get("company"),
            state.get("title"),
            state.get("needs_check"),
            state.get("next_action", "无"),
        )
        return {}  # 状态已包含所需信息

    # ── 执行入口 ──────────────────────────────────

    async def run(
        self,
        application: dict,
    ) -> dict[str, Any]:
        """执行投递跟踪工作流。

        Args:
            application: 投递记录字典（Application model_dump 或原始 dict）。

        Returns:
            包含处理结果的状态字典。
        """
        if _LANGGRAPH_AVAILABLE and self._graph is not None:
            # LangGraph 模式
            initial_state: ApplicationTrackingState = {
                "application": application,
                "company": "",
                "title": "",
                "status": "",
                "needs_check": False,
                "check_result": None,
                "next_action": None,
                "errors": [],
            }
            result = await self._graph.ainvoke(initial_state)
            return result
        else:
            # 回退模式：顺序执行
            return await self._run_fallback(application)

    async def _run_fallback(self, application: dict) -> dict[str, Any]:
        """无需 LangGraph 的回退顺序执行。"""
        state: dict[str, Any] = {
            "application": application,
            "company": application.get("company", ""),
            "title": application.get("title", ""),
            "status": application.get("status", ""),
            "needs_check": False,
            "check_result": None,
            "next_action": None,
            "errors": [],
        }

        # Step 1: load
        loaded = self._load_application(state)
        state.update(loaded)

        # Step 2: check
        checked = self._check_status(state)
        state.update(checked)

        # Step 3: route + suggest
        if state.get("needs_check"):
            suggested = self._generate_suggestion(state)
            state.update(suggested)

        # Step 4: output
        self._output_result(state)

        return state


# ═══════════════════════════════════════════════════════
# 2. 面试准备工作流
# ═══════════════════════════════════════════════════════

class InterviewPrepWorkflow:
    """面试准备工作流。

    步骤：
    1. 加载简历和 JD 文本
    2. 执行匹配分析（使用 RAGPipeline）
    3. 生成准备计划
    4. 生成每日回顾
    """

    def __init__(self) -> None:
        self._graph: Any = None
        self._build_graph()

    def _build_graph(self) -> None:
        """构建 LangGraph StateGraph。"""
        if not _LANGGRAPH_AVAILABLE:
            logger.info("LangGraph 不可用，工作流使用回退顺序模式")
            return

        workflow = StateGraph(InterviewPrepState)

        # 定义节点
        workflow.add_node("load_inputs", self._load_inputs)
        workflow.add_node("analyze_match", self._analyze_match)
        workflow.add_node("generate_plan", self._generate_plan)
        workflow.add_node("generate_review", self._generate_review)
        workflow.add_node("output_result", self._output_result)

        # 边
        workflow.set_entry_point("load_inputs")
        workflow.add_edge("load_inputs", "analyze_match")
        workflow.add_edge("analyze_match", "generate_plan")
        workflow.add_edge("generate_plan", "generate_review")
        workflow.add_edge("generate_review", "output_result")
        workflow.add_edge("output_result", END)

        self._graph = workflow.compile()

    # ── 节点函数 ──────────────────────────────────

    @staticmethod
    async def _load_inputs(state: InterviewPrepState) -> dict:
        """加载并验证输入。"""
        resume = state.get("resume_text", "")
        jd = state.get("jd_text", "")

        errors = []
        if not resume.strip():
            errors.append("简历文本为空")
        if not jd.strip():
            errors.append("JD 文本为空")

        if errors:
            logger.warning("输入验证失败: %s", errors)

        return {"errors": errors}

    @staticmethod
    async def _analyze_match(state: InterviewPrepState) -> dict:
        """执行简历-JD 匹配分析。"""
        if state.get("errors"):
            return {"match_result": None}

        try:
            # 构造简化的 resume_data 和 jd_data
            resume_data = {"raw_text": state.get("resume_text", "")}
            jd_data = {"raw_text": state.get("jd_text", "")}

            result = await preparation_service.analyze_match(resume_data, jd_data)

            match_dict = {
                "resume_name": result.resume_name,
                "jd_title": result.jd_title,
                "company": result.company,
                "match_score": result.match_score,
                "matched_skills": result.matched_skills,
                "missing_skills": result.missing_skills,
                "analysis": result.analysis,
            }

            logger.info(
                "匹配分析完成: %s - %s (%.2f)",
                result.company, result.jd_title, result.match_score,
            )

            return {"match_result": match_dict}

        except Exception as e:
            logger.error("匹配分析失败: %s", e, exc_info=True)
            return {"errors": [f"匹配分析失败: {e}"]}

    @staticmethod
    async def _generate_plan(state: InterviewPrepState) -> dict:
        """生成面试准备计划。"""
        if state.get("errors") or not state.get("match_result"):
            return {"prep_plan": None}

        match_result = state["match_result"]

        # 从 match_result dict 构造 MatchResult 对象
        mr = MatchResult(
            resume_name=match_result.get("resume_name", ""),
            jd_title=match_result.get("jd_title", ""),
            company=match_result.get("company", ""),
            match_score=match_result.get("match_score", 0.0),
            matched_skills=match_result.get("matched_skills", []),
            missing_skills=match_result.get("missing_skills", []),
            analysis=match_result.get("analysis", ""),
        )

        interview_date_str = state.get("interview_date")
        interview_date = date.fromisoformat(interview_date_str) if interview_date_str else date.today() + timedelta(days=7)

        try:
            plan = await preparation_service.generate_prep_plan(
                interview_date=interview_date,
                match_result=mr,
            )

            logger.info(
                "准备计划生成完成: %d 天计划",
                len(plan.get("daily_plans", [])),
            )

            return {"prep_plan": plan}

        except Exception as e:
            logger.error("准备计划生成失败: %s", e, exc_info=True)
            return {"errors": [f"准备计划生成失败: {e}"]}

    @staticmethod
    async def _generate_review(state: InterviewPrepState) -> dict:
        """生成每日回顾内容。"""
        if state.get("errors") or not state.get("prep_plan"):
            return {"daily_review": None}

        plan = state["prep_plan"]

        try:
            # 计算已过去的天数（从计划第一天算起）
            today = date.today()
            plan_date_str = None
            daily_plans = plan.get("daily_plans", [])
            if daily_plans:
                plan_date_str = daily_plans[0].get("date")

            day_offset = 0
            if plan_date_str:
                try:
                    plan_start = date.fromisoformat(plan_date_str)
                    day_offset = max(0, (today - plan_start).days)
                except ValueError:
                    pass

            review = await preparation_service.generate_daily_review(
                plan=plan,
                day_offset=day_offset,
            )

            return {"daily_review": review}

        except Exception as e:
            logger.error("每日回顾生成失败: %s", e, exc_info=True)
            return {"errors": [f"每日回顾生成失败: {e}"]}

    @staticmethod
    async def _output_result(state: InterviewPrepState) -> dict:
        """输出最终结果。"""
        logger.info(
            "面试准备工作流完成 | 匹配分数: %s | 计划天数: %d",
            state.get("match_result", {}).get("match_score", "N/A"),
            len(state.get("prep_plan", {}).get("daily_plans", [])),
        )
        return {}

    # ── 执行入口 ──────────────────────────────────

    async def run(
        self,
        resume_text: str,
        jd_text: str,
        interview_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行面试准备工作流。

        Args:
            resume_text: 简历全文。
            jd_text: JD 全文。
            interview_date: 面试日期（ISO 格式 YYYY-MM-DD），可选。

        Returns:
            包含 match_result, prep_plan, daily_review 的状态字典。
        """
        if _LANGGRAPH_AVAILABLE and self._graph is not None:
            # LangGraph 模式
            initial_state: InterviewPrepState = {
                "resume_text": resume_text,
                "jd_text": jd_text,
                "interview_date": interview_date,
                "match_result": None,
                "prep_plan": None,
                "daily_review": None,
                "errors": [],
            }
            result = await self._graph.ainvoke(initial_state)
            return result
        else:
            # 回退模式
            return await self._run_fallback(resume_text, jd_text, interview_date)

    async def _run_fallback(
        self,
        resume_text: str,
        jd_text: str,
        interview_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """无需 LangGraph 的回退顺序执行。"""
        state: InterviewPrepState = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "interview_date": interview_date,
            "match_result": None,
            "prep_plan": None,
            "daily_review": None,
            "errors": [],
        }

        state.update(await self._load_inputs(state))

        if not state.get("errors"):
            state.update(await self._analyze_match(state))

        if not state.get("errors") and state.get("match_result"):
            state.update(await self._generate_plan(state))

        if not state.get("errors") and state.get("prep_plan"):
            state.update(await self._generate_review(state))

        await self._output_result(state)

        return {
            "match_result": state.get("match_result"),
            "prep_plan": state.get("prep_plan"),
            "daily_review": state.get("daily_review"),
            "errors": state.get("errors", []),
        }


# ═══════════════════════════════════════════════════════
# 模块级实例
# ═══════════════════════════════════════════════════════

application_tracking_workflow = ApplicationTrackingWorkflow()
interview_prep_workflow = InterviewPrepWorkflow()
