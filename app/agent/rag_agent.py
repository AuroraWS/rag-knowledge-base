"""RAG 多智能体编排 — 查询改写 / 多步检索规划 / 答案合成与护栏。

LangGraph StateGraph 定义五个 Agent 节点的协作流程：
1. rewrite_query    — Query Rewriting Agent（模糊→精确，展开缩写，分解复合问题）
2. plan_retrieval   — Retrieval Planning Agent（决定子查询并行/串行执行策略）
3. execute_search   — 对每个子查询执行双路检索+Reranker，合并去重
4. synthesize       — Answer Synthesis Agent（基于上下文生成带引用标注的回答）
5. guardrails       — Guardrails Agent（安全检测+幻觉检测+引用完整性校验）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ── LangGraph 可用性检查 ───────────────────────────

try:
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph 未安装，RAG Agent 将使用回退顺序模式")

# ── 状态定义 ───────────────────────────────────────

if _LANGGRAPH_AVAILABLE:

    class RAGAgentState(TypedDict, total=False):
        original_question: str
        rewritten_question: str
        sub_queries: list[str]
        retrieval_results: list[dict[str, Any]]
        context: str
        draft_answer: str
        final_answer: str
        sources: list[dict[str, Any]]
        has_hallucination: bool
        safety_flagged: bool
        errors: list[str]
        latency_ms: float


# ── Agent 工作流 ────────────────────────────────────


class RAGAgentWorkflow:
    """RAG 多智能体工作流。

    5 个节点顺序执行（LangGraph 或顺序回退）：
    rewrite → plan → search → synthesize → guardrails
    """

    def __init__(self, pipeline: Any = None):
        if pipeline is None:
            from app.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline()
        self._pipeline = pipeline
        self._graph: Any = None
        self._build_graph()

    def _build_graph(self):
        if not _LANGGRAPH_AVAILABLE:
            return
        workflow = StateGraph(RAGAgentState)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("plan_retrieval", self._plan_retrieval)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("guardrails", self._guardrails)

        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "plan_retrieval")
        workflow.add_edge("plan_retrieval", "execute_search")
        workflow.add_edge("execute_search", "synthesize")
        workflow.add_edge("synthesize", "guardrails")
        workflow.add_edge("guardrails", END)

        self._graph = workflow.compile()

    # ── 节点: 查询改写 ──────────────────────────────

    async def _rewrite_query(self, state: RAGAgentState) -> dict:
        """Query Rewriting Agent: 改写模糊查询，展开缩写，分解复合问题。"""
        question = state.get("original_question", "")
        if not question:
            return {"errors": ["问题为空"]}

        prompt = f"""请改写以下用户查询，使其更适合知识库检索。要求：
1. 如果包含缩写，补充全称（例如：RAG → Retrieval Augmented Generation（检索增强生成））
2. 如果查询模糊，补充相关关键词以扩大检索范围
3. 如果包含多个子问题，分解为清晰的独立查询
4. 保持原始意图，不添加不存在的信息

原始查询：{question}

请按 JSON 格式输出：
{{
    "rewritten_query": "改写后的完整查询",
    "sub_queries": ["子查询1", "子查询2"],
    "expansion_note": "改写的说明"
}}"""

        try:
            result = await self._pipeline.generator.generate_structured(
                prompt=prompt,
                system_prompt="你是一个查询改写专家，擅长优化知识库检索查询。只输出 JSON。",
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("查询改写失败: %s，使用原始查询", e)
            return {
                "rewritten_question": question,
                "sub_queries": [question],
            }

        rewritten = result.get("rewritten_query", question) if isinstance(result, dict) else question
        sub_queries = result.get("sub_queries", [question]) if isinstance(result, dict) else [question]
        logger.info("查询改写: %s → %s (子查询: %d)", question, rewritten, len(sub_queries))
        return {"rewritten_question": rewritten, "sub_queries": sub_queries}

    # ── 节点: 检索规划 ──────────────────────────────

    async def _plan_retrieval(self, state: RAGAgentState) -> dict:
        """Retrieval Planning Agent: 决定子查询的执行策略（并行/串行）。"""
        sub_queries = state.get("sub_queries", [state.get("rewritten_question", "")])
        # 当前实现：所有子查询并行检索（后续可扩展为依赖图规划）
        return {"sub_queries": sub_queries}

    # ── 节点: 执行检索 ──────────────────────────────

    async def _execute_search(self, state: RAGAgentState) -> dict:
        """对每个子查询执行双路检索 + Reranker，合并去重。"""
        sub_queries = state.get("sub_queries", [state.get("rewritten_question", "")])
        all_results = []
        seen_texts = set()

        for sq in sub_queries:
            retrieved = self._pipeline.retriever.search(sq, top_k=settings.top_k)
            if not retrieved:
                continue

            docs_for_rerank = [
                {"text": text, "metadata": meta, "retrieval_score": score}
                for text, meta, score in retrieved
            ]
            reranked = self._pipeline.reranker.rerank_with_docs(
                sq, docs_for_rerank, top_k=settings.rerank_top_k
            )

            for doc in reranked:
                text = doc["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(doc)

        context_parts = []
        sources = []
        for i, doc in enumerate(all_results):
            context_parts.append(f"[来源 {i + 1}] {doc['text']}")
            sources.append({
                "content": doc["text"][:500],
                "score": doc.get("rerank_score", 0),
                "source": doc.get("metadata", {}).get("source", f"doc_{i}"),
            })

        context = "\n\n".join(context_parts)
        logger.info("检索完成: %d 个唯一结果", len(all_results))

        return {
            "retrieval_results": all_results,
            "context": context,
            "sources": sources,
        }

    # ── 节点: 答案合成 ──────────────────────────────

    async def _synthesize(self, state: RAGAgentState) -> dict:
        """Answer Synthesis Agent: 基于检索上下文生成带引用标注的回答。"""
        context = state.get("context", "")
        question = state.get("rewritten_question", state.get("original_question", ""))

        if not context:
            # 无上下文时直接 LLM 回答
            answer = await self._pipeline.generator.generate(
                prompt=question,
                system_prompt="请直接回答用户的问题。",
            )
            return {"draft_answer": answer}

        prompt = f"""基于以下参考信息，回答用户的问题。

参考信息：
{context}

用户问题：{question}

请基于参考信息给出准确、详细的回答。如果参考信息不足以回答问题，请明确说明。
回答中请使用 [来源 X] 标注引用自哪个参考来源。"""

        answer = await self._pipeline.generator.generate(
            prompt=prompt,
            system_prompt="你是一个专业的知识库问答助手。必须基于提供的参考信息回答，无法回答时明确说明。使用 [来源 X] 标注引用。",
        )
        return {"draft_answer": answer}

    # ── 节点: 护栏 ──────────────────────────────────

    async def _guardrails(self, state: RAGAgentState) -> dict:
        """Guardrails Agent: 安全检测 + 幻觉检测 + 引用完整性校验。"""
        draft = state.get("draft_answer", "")
        sources = state.get("sources", [])
        context = state.get("context", "")

        if not draft:
            return {"final_answer": "", "has_hallucination": False, "safety_flagged": False}

        guardrails_prompt = f"""请对以下回答进行质量检查，按 JSON 格式输出。

原始问题：{state.get('original_question', '')}

参考来源：
{context[:2000] if context else '无参考来源'}

回答草案：
{draft}

请检查：
1. 安全性：回答是否包含任何不当、有害或歧视性内容？
2. 真实性：回答中的关键事实是否都有参考来源支持？是否有编造的内容？
3. 引用完整性：引用标记（[来源 X]）是否都指向了实际存在的参考来源？

输出 JSON：
{{
    "safety_flagged": false,
    "safety_reason": "",
    "hallucination_detected": false,
    "hallucination_details": "",
    "citation_issues": [],
    "final_answer": "修正后的最终回答（保持原回答或做必要修正）"
}}"""

        try:
            result = await self._pipeline.generator.generate_structured(
                prompt=guardrails_prompt,
                system_prompt="你是一个回答质量审核专家，负责检查回答的安全性和真实性。只输出 JSON。",
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("护栏检查失败: %s，使用原始回答", e)
            return {
                "final_answer": draft,
                "has_hallucination": False,
                "safety_flagged": False,
            }

        final = result.get("final_answer", draft) if isinstance(result, dict) else draft
        has_hallucination = result.get("hallucination_detected", False) if isinstance(result, dict) else False
        safety_flagged = result.get("safety_flagged", False) if isinstance(result, dict) else False

        if has_hallucination:
            logger.warning("检测到幻觉: %s", result.get("hallucination_details", ""))
        if safety_flagged:
            logger.warning("安全问题: %s", result.get("safety_reason", ""))

        return {
            "final_answer": final,
            "has_hallucination": has_hallucination,
            "safety_flagged": safety_flagged,
        }

    # ── 执行入口 ────────────────────────────────────

    async def run(self, question: str) -> dict[str, Any]:
        """执行完整的多智能体 RAG 工作流。"""
        start = time.time()

        if _LANGGRAPH_AVAILABLE and self._graph is not None:
            initial: RAGAgentState = {
                "original_question": question,
                "rewritten_question": "",
                "sub_queries": [],
                "retrieval_results": [],
                "context": "",
                "draft_answer": "",
                "final_answer": "",
                "sources": [],
                "has_hallucination": False,
                "safety_flagged": False,
                "errors": [],
                "latency_ms": 0.0,
            }
            result = await self._graph.ainvoke(initial)
        else:
            result = await self._run_fallback(question)

        elapsed = (time.time() - start) * 1000
        result["latency_ms"] = round(elapsed, 2)
        return result

    async def _run_fallback(self, question: str) -> dict[str, Any]:
        """无 LangGraph 时的顺序回退执行。"""
        state: dict[str, Any] = {
            "original_question": question,
            "rewritten_question": "",
            "sub_queries": [],
            "retrieval_results": [],
            "context": "",
            "draft_answer": "",
            "final_answer": "",
            "sources": [],
            "has_hallucination": False,
            "safety_flagged": False,
            "errors": [],
            "latency_ms": 0.0,
        }

        state.update(await self._rewrite_query(state))
        state.update(await self._plan_retrieval(state))
        state.update(await self._execute_search(state))
        state.update(await self._synthesize(state))
        state.update(await self._guardrails(state))

        return state


# 模块级单例
rag_agent_workflow = RAGAgentWorkflow()
