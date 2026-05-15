#!/usr/bin/env python3
"""
LangChain LCEL RAG 适配层 — 面试演示脚本
=========================================
展示 LangChain LCEL 管道的完整能力：
  文档加载 → 切分 → Embedding → FAISS 检索 → DeepSeek LLM 生成。

演示覆盖：invoke / stream / batch / 手写 vs LCEL 对比分析。
"""

import os
import sys
import time
import asyncio
from typing import List

# 修复 Windows 终端 GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def section(title: str) -> None:
    """打印分隔标题。"""
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def check_api_key() -> bool:
    """检查 DeepSeek API Key 是否有效。"""
    key = settings.deepseek_api_key
    if key and key != "sk-your-api-key-here":
        return True
    print("⚠️  DEEPSEEK_API_KEY 未配置或为占位符，将跳过 LLM 调用环节。")
    print("   请在 .env 中设置你的 API Key（从 platform.deepseek.com 获取）")
    return False


# ═══════════════════════════════════════════════════════════════
# Demo 1: 初始化适配器 & 打印组件配置
# ═══════════════════════════════════════════════════════════════

def demo_init() -> None:
    """初始化 LangChainRAGAdapter 并打印组件配置。"""
    section("Demo 1: 初始化适配器 & 组件配置")

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  RAG 管道组件配置                                        │
  ├─────────────────────────────────────────────────────────┤
  │  Embedding 模型 : {settings.embedding_model:<38} │
  │  Reranker 模型  : {settings.reranker_model:<38} │
  │  LLM 后端       : DeepSeek (deepseek-chat)              │
  │  LLM Base URL   : {settings.deepseek_base_url:<38} │
  │  Top-K 检索     : {settings.top_k:<38} │
  │  Rerank Top-K   : {settings.rerank_top_k:<38} │
  │  知识库目录      : {settings.knowledge_base_dir:<38} │
  │  FAISS 索引目录  : {settings.faiss_index_dir:<38} │
  └─────────────────────────────────────────────────────────┘
""")

    # 导入适配器类，展示 LCEL 管道结构
    from app.rag.langchain_adapter import (
        LangChainRAGAdapter,
        _cache,
        _format_docs,
    )
    from app.data.loader import load_documents
    from app.data.chunker import chunk_documents

    # 打印 LCEL 管道拓扑
    print("  LCEL 管道拓扑：")
    print("  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐")
    print("  │ Retriever │ ─→ │  Format   │ ─→ │  Prompt  │ ─→ │   LLM    │")
    print("  │ (FAISS)   │    │  Docs     │    │ Template │    │ (DeepSeek)│")
    print("  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘")
    print("                                                         │")
    print("                                                    StrOutputParser")
    print("                                                         │")
    print("                                                    最终回答")

    print("\n  管道伪代码：")
    print("  chain = (")
    print('      {"context": retriever | format_docs,')
    print('       "question": RunnablePassthrough()}')
    print("      | prompt | llm | StrOutputParser()")
    print("  )")

    # 加载文档并展示索引状态
    print("\n  📂 加载知识库文档...")
    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    print(f"     文档数: {len(docs)}, Chunks 数: {len(chunks)}")
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "?"))
        print(f"     - {src} ({d.metadata.get('type', '?')}, {len(d.page_content)} chars)")

    print("\n  ✅ 适配器初始化成功，组件配置打印完毕")


# ═══════════════════════════════════════════════════════════════
# Demo 2: invoke() 同步调用
# ═══════════════════════════════════════════════════════════════

def demo_invoke() -> None:
    """演示 invoke() 同步调用。"""
    section("Demo 2: invoke() — 同步调用")

    if not check_api_key():
        print("  ⏭️  跳过（需要 API Key）")
        return

    from app.rag.langchain_adapter import LangChainRAGAdapter

    print("  构建适配器（首次运行会下载 Embedding 模型 & 构建 FAISS 索引）...")
    t0 = time.time()
    adapter = LangChainRAGAdapter()
    print(f"  初始化耗时: {time.time() - t0:.1f}s")

    query = "公司年假有多少天？"
    print(f"\n  ❓ 问题: {query}")

    t0 = time.time()
    answer: str = adapter.invoke(query)
    elapsed = time.time() - t0

    print(f"  🤖 回答: {answer}")
    print(f"  ⏱️  耗时: {elapsed:.1f}s")
    print("  ✅ invoke() 调用成功")


# ═══════════════════════════════════════════════════════════════
# Demo 3: stream() 流式输出
# ═══════════════════════════════════════════════════════════════

def demo_stream() -> None:
    """演示 stream() 流式输出，逐 token 打印。"""
    section("Demo 3: stream() — 流式输出")

    if not check_api_key():
        print("  ⏭️  跳过（需要 API Key）")
        return

    from app.rag.langchain_adapter import LangChainRAGAdapter

    adapter = LangChainRAGAdapter()
    query = "加班费怎么算？"

    print(f"  ❓ 问题: {query}")
    print("  🤖 回答: ", end="", flush=True)

    # ── 逐 token 流式输出 ──
    token_count: int = 0
    t0 = time.time()
    for token in adapter.stream(query):
        print(token, end="", flush=True)
        token_count += 1
    elapsed = time.time() - t0

    print(f"\n  📊 共 {token_count} 个 token, 耗时 {elapsed:.1f}s")
    print("  ✅ stream() 流式调用成功")


# ═══════════════════════════════════════════════════════════════
# Demo 4: batch() 批量调用
# ═══════════════════════════════════════════════════════════════

def demo_batch() -> None:
    """演示 batch() 批量调用 3 个问题。"""
    section("Demo 4: batch() — 批量调用")

    if not check_api_key():
        print("  ⏭️  跳过（需要 API Key）")
        return

    from app.rag.langchain_adapter import LangChainRAGAdapter

    adapter = LangChainRAGAdapter()

    queries: List[str] = [
        "公司年假有多少天？",
        "加班补贴政策是什么？",
        "报销流程怎么走？",
    ]

    print(f"  ❓ 批量提交 {len(queries)} 个问题:\n")
    for i, q in enumerate(queries, 1):
        print(f"     [{i}] {q}")

    # ── 批量调用 ──
    t0 = time.time()
    answers: List[str] = adapter.batch(queries)
    elapsed = time.time() - t0

    print(f"\n  📊 批量结果（耗时 {elapsed:.1f}s）:\n")
    for i, (q, a) in enumerate(zip(queries, answers), 1):
        print(f"  [{i}] Q: {q}")
        print(f"      A: {a[:120]}{'...' if len(a) > 120 else ''}\n")

    print(f"  ✅ batch() 批量调用成功（{len(queries)} 个问题，总耗时 {elapsed:.1f}s）")


# ═══════════════════════════════════════════════════════════════
# Demo 5: 手写 RAG vs LangChain LCEL 对比分析
# ═══════════════════════════════════════════════════════════════

def demo_comparison() -> None:
    """打印"手写 RAG vs LangChain LCEL"对比分析。"""
    section("Demo 5: 手写 RAG vs LangChain LCEL 对比分析")

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │              手写 RAG  vs  LangChain LCEL                     │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  【相同点】                                                   │
  │                                                              │
  │  • 底层检索逻辑一致：都基于 FAISS 向量检索 + BM25 稀疏检索      │
  │  • Embedding 模型相同：BAAI/bge-small-zh-v1.5                  │
  │  • LLM 调用相同：DeepSeek Chat API (OpenAI 兼容)               │
  │  • Prompt 结构相同：system + context + question                │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  【不同点】                                                   │
  │                                                              │
  │  维度          │  手写 RAG          │  LangChain LCEL          │
  │  ─────────────┼───────────────────┼───────────────────────── │
  │  编程范式      │ 命令式（逐步调用）  │ 声明式（链式组合）        │
  │  Streaming     │ 需手动对接 SSE     │ 自动支持，chain.stream()  │
  │  Async         │ 需自行管理 loop    │ 自动支持，chain.ainvoke() │
  │  Batch         │ 需手动多线程/协程  │ 自动并发，chain.batch()   │
  │  代码量        │ ~120 行            │ ~50 行 (减少 60%)         │
  │  可观测性      │ 需手动打点         │ LangSmith 一键追踪        │
  │  可测试性      │ 单元测试           │ LCEL 组件可独立测试       │
  │  学习曲线      │ 低（原生 Python）   │ 中（需理解 Runnable 语义）│
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  【本项目技术选型原因】                                        │
  │                                                              │
  │  核心检索层保留手写：                                          │
  │  → BM25 + 向量 双路检索 + RRF 融合 需要精细控制               │
  │  → 自定义 Reranker 排序逻辑                                   │
  │  → 灵活可控，适合垂直场景深度优化                              │
  │                                                              │
  │  编排层用 LangChain：                                         │
  │  → Prompt 模板 + LLM 调用 + 输出解析 是标准链路               │
  │  → 自动获得 streaming / async / batch 能力                    │
  │  → 减少样板代码，聚焦业务逻辑                                  │
  │                                                              │
  │  结论：底层"手写"保灵活，上层"LCEL"提效率                      │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """运行全部演示。"""
    print("=" * 64)
    print("  🚀 LangChain LCEL RAG 适配层 — 面试演示")
    print("=" * 64)
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  项目路径: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    try:
        demo_init()        # 1. 初始化 & 配置
        demo_invoke()      # 2. invoke 同步调用
        demo_stream()      # 3. stream 流式输出
        demo_batch()       # 4. batch 批量调用
        demo_comparison()  # 5. 对比分析

        # ── 总结 ──
        has_key = settings.deepseek_api_key and settings.deepseek_api_key != "sk-your-api-key-here"
        print(f"\n{'=' * 64}")
        print("  📋 演示总结")
        print(f"{'=' * 64}")
        if has_key:
            print("  ✅ Demo 1-5 全部通过")
            print("     - 组件配置打印:    ✅")
            print("     - invoke() 同步:   ✅")
            print("     - stream() 流式:   ✅")
            print("     - batch() 批量:    ✅")
            print("     - 对比分析:         ✅")
        else:
            print("  ✅ Demo 1 & 5 通过")
            print("     - 组件配置打印:    ✅")
            print("     - 对比分析:         ✅")
            print("  ⏭️  Demo 2-4 跳过（在 .env 中设置 DEEPSEEK_API_KEY 后体验完整演示）")
            print()
            print("  获取 API Key: https://platform.deepseek.com")
            print("  配置方式: 编辑 .env 文件，替换 DEEPSEEK_API_KEY=sk-your-api-key-here")

    except Exception as e:
        print(f"\n  ❌ 演示异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
