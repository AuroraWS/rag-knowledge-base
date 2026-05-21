"""
RAG 流水线编排 — 连接 Embedding → Retriever → Reranker → Generator

还包含简历-JD 智能匹配逻辑：
利用 RAG 检索相关技能/经验片段 + LLM 综合分析与评分。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from app.config import settings
from app.models.schema import MatchResult
from app.rag.embeddings import EmbeddingModel
from app.rag.generator import LLMGenerator
from app.rag.reranker import Reranker
from app.rag.retriever import DualRetriever

logger = logging.getLogger(__name__)

# 简历-JD 匹配的系统提示
MATCH_SYSTEM_PROMPT = """你是一位专业的AI招聘顾问，擅长分析简历与岗位描述(JD)的匹配度。
你的任务是对简历和JD进行深度对比分析，并给出JSON格式的匹配结果。

请严格按照以下JSON格式输出（不要包含markdown代码块标记）：
{
    "match_score": 0.85,
    "matched_skills": ["Python", "PyTorch", ...],
    "missing_skills": ["Kubernetes", ...],
    "analysis": "详细的匹配分析...",
    "suggestions": ["建议1", "建议2", ...]
}

评分标准：
- match_score: 0~1之间的浮点数
  - 0.9~1.0: 高度匹配，几乎不需要调整
  - 0.7~0.9: 良好匹配，有少量差距
  - 0.5~0.7: 部分匹配，有较明显差距
  - 0.3~0.5: 匹配度较低，需要大量调整
  - 0.0~0.3: 基本不匹配
- matched_skills: 简历中完全符合JD要求的技能列表
- missing_skills: JD要求但简历中未体现的技能列表
- analysis: 详细的匹配分析，包括各维度的评估（技术栈、工作经验、项目经历等）
- suggestions: 针对缺失项的改进建议
"""


class RAGPipeline:
    """RAG 完整流水线 — 编排检索、重排序、生成全流程"""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        retriever: Optional[DualRetriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[LLMGenerator] = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.retriever = retriever or DualRetriever(embedding_model=self.embedding_model)
        self.reranker = reranker or Reranker()
        self.generator = generator or LLMGenerator()

    # ── 文档管理 ─────────────────────────────────────────

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        """添加一篇文档到检索器"""
        self.retriever.add_documents([{"text": text, "metadata": metadata or {}}])

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """批量添加文档。每个 dict 需含 'text' 键，可选 'metadata'"""
        self.retriever.add_documents(docs)

    def build_index(self) -> None:
        """构建检索索引（添加文档后需调用此方法）"""
        self.retriever.build_index()

    # ── RAG 查询 ────────────────────────────────────────

    async def query(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_k: int = 3,
        system_prompt: str = "",
        return_raw_sources: bool = False,
    ) -> dict[str, Any]:
        """
        执行完整 RAG 查询。

        Parameters
        ----------
        question : str
            用户问题
        top_k : int
            召回阶段取 top_k 文档
        rerank_top_k : int
            重排序后取 rerank_top_k 文档
        system_prompt : str
            可选的系统提示词
        return_raw_sources : bool
            是否返回原始检索结果（含分数）

        Returns
        -------
        dict
            {
                "answer": str,           # 生成的回答
                "sources": list[dict],   # 参考来源
                "latency_ms": float,     # 总耗时（毫秒）
            }
        """
        start = time.time()

        # ── 1. 检索 ─────────────────────────────────
        retrieved = self.retriever.search(question, top_k=top_k)
        if not retrieved:
            logger.warning("未检索到相关文档，使用 LLM 直接回答")
            answer = await self.generator.generate(
                prompt=question,
                system_prompt=system_prompt or "请直接回答用户的问题。",
            )
            elapsed = (time.time() - start) * 1000
            return {
                "answer": answer,
                "sources": [],
                "latency_ms": round(elapsed, 2),
            }

        # ── 2. 重排序 ───────────────────────────────
        docs_for_rerank = [
            {"text": text, "metadata": meta, "retrieval_score": score}
            for text, meta, score in retrieved
        ]
        reranked = self.reranker.rerank_with_docs(
            question, docs_for_rerank, top_k=rerank_top_k
        )

        # ── 3. 拼上下文 ─────────────────────────────
        context_parts = []
        sources = []
        for i, doc in enumerate(reranked):
            text = doc["text"]
            meta = doc.get("metadata", {})
            source_info = {
                "content": text[:500],  # 截断过长内容
                "score": round(doc.get("rerank_score", 0), 4),
                "source": meta.get("source", meta.get("title", f"doc_{i}")),
            }
            if return_raw_sources:
                source_info["metadata"] = meta
            sources.append(source_info)
            context_parts.append(f"[来源 {i+1}] {text}")

        context = "\n\n".join(context_parts)

        # ── 4. 生成 ─────────────────────────────────
        prompt = f"""基于以下参考信息，回答用户的问题。

参考信息：
{context}

用户问题：{question}

请基于参考信息给出准确、详细的回答。如果参考信息不足以回答问题，请说明。"""

        final_system = system_prompt or "你是一个专业的知识库问答助手，请基于提供的信息准确回答。"
        answer = await self.generator.generate(prompt=prompt, system_prompt=final_system)

        elapsed = (time.time() - start) * 1000
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(elapsed, 2),
        }

    # ── 简历-JD 匹配 ────────────────────────────────────

    async def match_resume_jd(
        self,
        resume_text: str,
        jd_text: str,
        resume_model: Any = None,
        jd_model: Any = None,
    ) -> dict[str, Any]:
        """
        简历与岗位描述(JD)的深度匹配分析。

        流程：
        1. 将简历和 JD 文本添加到检索器（如有已构建的索引则检索相关片段）
        2. 用 LLM 进行结构化匹配分析
        3. 返回匹配结果（分数、匹配/缺失技能、分析、建议）

        Parameters
        ----------
        resume_text : str
            简历全文
        jd_text : str
            JD 全文
        resume_model : Any, optional
            可选的结构化简历模型 (Resume Pydantic model)
        jd_model : Any, optional
            可选的结构化 JD 模型 (JobDescription Pydantic model)

        Returns
        -------
        dict
            {
                "match_score": float,
                "matched_skills": list[str],
                "missing_skills": list[str],
                "analysis": str,
                "suggestions": list[str],
                "latency_ms": float,
            }
        """
        start = time.time()

        # ── Step 1: 如果有结构化模型，提取关键字段增强 prompt ──
        resume_enhanced = resume_text
        jd_enhanced = jd_text

        if resume_model is not None:
            # 从简历模型中提取结构化信息
            skills_str = ", ".join(resume_model.skills) if resume_model.skills else "未提供"
            edu_str = "; ".join(
                [f"{e.school} {e.degree} {e.major}" for e in (resume_model.education or [])]
            )
            exp_str = "; ".join(
                [f"{e.company} {e.title}" for e in (resume_model.work_experience or [])]
            )
            resume_enhanced = f"""【原始简历】
{resume_text}

【结构化信息】
技能: {skills_str}
教育: {edu_str}
工作经历: {exp_str}"""

        if jd_model is not None:
            req_str = "; ".join(
                [f"[{r.category}] {r.content}" for r in (jd_model.requirements or [])]
            )
            resp_str = "; ".join(jd_model.responsibilities or [])
            jd_enhanced = f"""【原始JD】
{jd_text}

【结构化信息】
公司: {jd_model.company}
岗位: {jd_model.title}
地点: {jd_model.location}
职责: {resp_str}
要求: {req_str}"""

        # ── Step 2: 使用 RAG 检索相关片段 ──────────────
        # 如果检索器中已有文档，利用它们找到相关技能/经验描述
        context_fragments = []
        if self.retriever.document_count > 0:
            # 用 JD 文本作为 query 检索相关简历片段
            jd_results = self.retriever.search(jd_text, top_k=5)
            for text, meta, score in jd_results:
                context_fragments.append(f"[相关经验] {text[:300]}")

        context_str = "\n\n".join(context_fragments) if context_fragments else ""

        # ── Step 3: 构建匹配 prompt ─────────────────
        match_prompt = f"""请对以下简历和岗位描述(JD)进行详细的匹配分析。

=== 简历 ===
{resume_enhanced}

=== 岗位描述(JD) ===
{jd_enhanced}
"""
        if context_str:
            match_prompt += f"\n\n=== 检索到的相关参考片段 ===\n{context_str}\n"

        match_prompt += """
请分析以上简历与JD的匹配程度，输出JSON格式（不要包含markdown代码块标记）：
{
    "match_score": 0.85,
    "matched_skills": ["技能1", "技能2"],
    "missing_skills": ["缺失技能1", "缺失技能2"],
    "analysis": "详细分析...",
    "suggestions": ["建议1", "建议2"]
}"""

        # ── Step 4: LLM 调用 ────────────────────────
        try:
            result = await self.generator.generate_structured(
                prompt=match_prompt,
                system_prompt=MATCH_SYSTEM_PROMPT,
                temperature=0.1,  # 结构化输出用低温度
            )

            # 如果返回了字符串（非预期），尝试解析 JSON
            if isinstance(result, str):
                result = json.loads(result)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("LLM 匹配结果解析失败: %s", e)
            # 降级：返回一个基础结果
            result = {
                "match_score": 0.5,
                "matched_skills": [],
                "missing_skills": [],
                "analysis": f"匹配分析生成失败: {e}",
                "suggestions": ["请重新提交或手动分析"],
            }

        elapsed = (time.time() - start) * 1000
        result["latency_ms"] = round(elapsed, 2)

        return result

    async def match_resume_jd_structured(
        self,
        resume_text: str,
        jd_text: str,
        resume_model: Any = None,
        jd_model: Any = None,
    ) -> MatchResult:
        """
        简历-JD 匹配，返回 MatchResult Pydantic 模型。

        与 match_resume_jd 类似但返回结构化对象。
        """
        raw = await self.match_resume_jd(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_model=resume_model,
            jd_model=jd_model,
        )

        return MatchResult(
            resume_name=getattr(resume_model, "name", "未知") if resume_model else "未知",
            jd_title=getattr(jd_model, "title", "未知") if jd_model else "未知",
            company=getattr(jd_model, "company", "未知") if jd_model else "未知",
            match_score=raw.get("match_score", 0.0),
            matched_skills=raw.get("matched_skills", []),
            missing_skills=raw.get("missing_skills", []),
            analysis=raw.get("analysis", ""),
        )

    # ── 清理 ─────────────────────────────────────────

    async def close(self) -> None:
        """释放资源"""
        await self.generator.close()

    async def __aenter__(self) -> "RAGPipeline":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"<RAGPipeline retriever={self.retriever.document_count}docs "
            f"reranker={self.reranker.model_name} "
            f"generator={self.generator}>"
        )
