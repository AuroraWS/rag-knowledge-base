"""智能招聘助手 v2.0 — Gradio 多页面前端

5 个功能标签页（资料库、投递管理、生成工具、准备建议、配置）。
通过 HTTP 调用 FastAPI 后端接口。"""

from __future__ import annotations

import json
import os

import gradio as gr
import httpx

from typing import Any

# ── 后端地址（可通过环境变量覆盖） ────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


# ── 通用 HTTP 工具函数 ────────────────────────────


def _get(path: str) -> dict:
    """向后端发送 GET 请求，返回 JSON 结果。"""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(f"{BACKEND_URL}{path}")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接后端 ({BACKEND_URL})，请确认后端已启动: uvicorn app.main:app --reload")


def _post(path: str, payload: dict | None) -> dict:
    """向后端发送 POST 请求，返回 JSON 结果。"""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BACKEND_URL}{path}", json=payload or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接后端 ({BACKEND_URL})，请确认后端已启动: uvicorn app.main:app --reload")


def _put(path: str, payload: dict | None | None) -> dict:
    """向后端发送 PUT 请求，返回 JSON 结果。"""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.put(f"{BACKEND_URL}{path}", json=payload or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接后端 ({BACKEND_URL})，请确认后端已启动: uvicorn app.main:app --reload")


def _delete(path: str) -> dict:
    """向后端发送 DELETE 请求，返回 JSON 结果。"""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.delete(f"{BACKEND_URL}{path}")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接后端 ({BACKEND_URL})，请确认后端已启动: uvicorn app.main:app --reload")


# ═══════════════════════════════════════════════════════
# Tab 1: 资料库 (Profile)
# ═══════════════════════════════════════════════════════


def load_profile() -> str:
    """加载完整简历信息。"""
    try:
        data = _get("/api/profile")
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"加载失败: {e}"


def upload_doc(file: Any) -> str:
    """上传简历文件。"""
    if file is None:
        return "请先选择文件"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with open(file.name, "rb") as f:
                resp = client.post(
                    f"{BACKEND_URL}/api/profile/upload",
                    files={"file": (os.path.basename(file.name), f)},
                )
                resp.raise_for_status()
                return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return f"后端返回格式异常: {e}\n请查看后端终端日志"
    except Exception as e:
        return f"上传失败: {e}"


def build_profile_tab():
    """构建资料库标签页。"""
    gr.Markdown("## 📄 资料库 — 个人简历信息")
    gr.Markdown("查看和更新您的个人基本信息、教育经历、工作经历、项目经历和证书。")

    with gr.Row():
        refresh_btn = gr.Button("🔄 刷新简历", variant="primary")
        upload_btn = gr.UploadButton("📤 上传简历文件", file_types=[".pdf", ".docx", ".doc", ".png", ".jpg"])

    profile_output = gr.JSON(label="简历数据 (JSON)")

    refresh_btn.click(fn=load_profile, outputs=profile_output)
    upload_btn.upload(fn=upload_doc, inputs=upload_btn, outputs=profile_output)


# ═══════════════════════════════════════════════════════
# Tab 2: 投递管理 (Applications)
# ═══════════════════════════════════════════════════════


def list_apps(status_filter: str = "") -> gr.Dataframe:
    """获取投递记录列表。"""
    try:
        suffix = f"?status={status_filter}" if status_filter else ""
        data = _get(f"/api/applications{suffix}")
        apps = data.get("applications", [])
        if not apps:
            return gr.Dataframe(value=[], headers=["公司", "岗位", "地点", "渠道", "状态", "投递日期"])

        rows = []
        for a in apps:
            rows.append([
                a.get("company", ""),
                a.get("title", ""),
                a.get("location", ""),
                a.get("channel", ""),
                a.get("status", ""),
                str(a.get("submit_date", "")),
            ])
        return gr.Dataframe(
            value=rows,
            headers=["公司", "岗位", "地点", "渠道", "状态", "投递日期"],
            label="投递记录",
        )
    except Exception as e:
        return gr.Dataframe(
            value=[[f"加载失败: {e}"]],
            headers=["错误"],
        )


def add_application(company: str, title: str, location: str, channel: str, url: str, jd_text: str) -> str:
    """添加新的投递记录。"""
    if not company or not title:
        return "公司名称和岗位名称为必填项"
    try:
        result = _post("/api/applications", {
            "company": company,
            "title": title,
            "location": location,
            "channel": channel or "手动录入",
            "url": url or "",
            "jd_text": jd_text or "",
        })
        return f"✅ 已添加: {result.get('data', {}).get('company', '')} - {result.get('data', {}).get('title', '')}"
    except Exception as e:
        return f"添加失败: {e}"


def update_app_status(app_id: str, new_status: str, note: str) -> str:
    """更新投递记录状态。"""
    if not app_id:
        return "请输入投递记录 ID"
    if not new_status:
        return "请选择新状态"
    try:
        _put(f"/api/applications/{app_id}/status", {
            "status": new_status,
            "note": note or "",
        })
        return f"✅ 状态已更新为: {new_status}"
    except Exception as e:
        return f"更新失败: {e}"


STATUS_CHOICES = ["待投递", "已投递待反馈", "已收到笔试", "面试中", "已拒绝", "已拿到Offer"]


def execute_command(text: str) -> str:
    """通过自然语言指令操作投递记录。"""
    if not text.strip():
        return "请输入指令，例如：我投了JPMorgan的Software Engineer"
    try:
        result = _post("/api/command", {"text": text})
        summary = result.get("action_summary", "")
        intent = result.get("intent", "")
        data = result.get("result")

        lines = [f"意图: {intent}", f"{summary}"]
        if data and isinstance(data, dict):
            if intent in ("add_application",):
                lines.append(f"公司: {data.get('company', '')}")
                lines.append(f"岗位: {data.get('title', '')}")
                lines.append(f"状态: {data.get('status', '')}")
            elif intent == "update_status":
                lines.append(f"公司: {data.get('company', '')} - {data.get('title', '')}")
                lines.append(f"新状态: {data.get('status', '')}")
            elif intent == "generate_self_intro":
                lines.append(data.get("intro", "")[:500])
        elif data and isinstance(data, list):
            lines.append(f"共 {len(data)} 条记录")
        return "\n".join(lines)
    except Exception as e:
        return f"指令执行失败: {e}"


def build_applications_tab():
    """构建投递管理标签页。"""
    gr.Markdown("## 📋 投递管理 — 投递记录跟踪")
    gr.Markdown("查看所有投递记录，添加新记录，更新状态。")

    # ── 筛选 + 刷新 ──
    with gr.Row():
        status_filter = gr.Dropdown(
            choices=[""] + STATUS_CHOICES,
            label="按状态筛选",
            value="",
        )
        refresh_btn = gr.Button("🔄 刷新列表", variant="primary")

    apps_table = gr.Dataframe(
        headers=["公司", "岗位", "地点", "渠道", "状态", "投递日期"],
        label="投递记录",
        row_count=(10, "dynamic"),
        interactive=False,
    )

    refresh_btn.click(fn=list_apps, inputs=status_filter, outputs=apps_table)
    status_filter.change(fn=list_apps, inputs=status_filter, outputs=apps_table)

    # ── 自然语言指令输入 ──
    gr.Markdown("---")
    gr.Markdown("### 💬 自然语言指令（推荐）")
    gr.Markdown(
        "直接输入指令，系统自动识别并执行。示例："
        "_\"我投了JPMorgan的Software Engineer\"_ · "
        "_\"字节跳动约了6月1号面试\"_ · "
        "_\"面试中的有哪些\"_"
    )
    with gr.Row():
        nl_input = gr.Textbox(
            label="输入指令",
            placeholder="例如：我投了JPMorgan的Software Engineer (2107374)",
            lines=2,
            scale=5,
        )
        nl_btn = gr.Button("🚀 执行", variant="primary", scale=1)
    nl_output = gr.Textbox(label="执行结果", interactive=False, lines=3)

    nl_btn.click(
        fn=execute_command,
        inputs=nl_input,
        outputs=nl_output,
    ).then(fn=list_apps, inputs=status_filter, outputs=apps_table)

    gr.Markdown("---")
    gr.Markdown("### ➕ 添加投递记录（手动表单）")

    with gr.Row():
        company_input = gr.Textbox(label="公司名称", scale=1)
        title_input = gr.Textbox(label="岗位名称", scale=1)
        location_input = gr.Textbox(label="工作地点", scale=1)
    with gr.Row():
        channel_input = gr.Textbox(label="投递渠道", value="手动录入", scale=1)
        url_input = gr.Textbox(label="投递链接", scale=1)
    jd_input = gr.Textbox(label="JD 全文", lines=3)
    add_btn = gr.Button("✅ 添加", variant="primary")
    add_output = gr.Textbox(label="结果", interactive=False)

    add_btn.click(
        fn=add_application,
        inputs=[company_input, title_input, location_input, channel_input, url_input, jd_input],
        outputs=add_output,
    ).then(fn=list_apps, inputs=status_filter, outputs=apps_table)

    gr.Markdown("---")
    gr.Markdown("### 🔄 更新投递状态")

    with gr.Row():
        app_id_input = gr.Textbox(label="投递记录 ID", scale=1)
        status_select = gr.Dropdown(
            choices=STATUS_CHOICES,
            label="新状态",
            value=STATUS_CHOICES[0],
            scale=1,
        )
    status_note = gr.Textbox(label="备注（可选）", lines=1)
    update_btn = gr.Button("🔄 更新状态", variant="secondary")
    update_output = gr.Textbox(label="结果", interactive=False)

    update_btn.click(
        fn=update_app_status,
        inputs=[app_id_input, status_select, status_note],
        outputs=update_output,
    ).then(fn=list_apps, inputs=status_filter, outputs=apps_table)


# ═══════════════════════════════════════════════════════
# Tab 3: 生成工具 (Generate)
# ═══════════════════════════════════════════════════════


def generate_self_intro(jd_text: str, style: str) -> str:
    """生成三段式自我介绍。"""
    if not jd_text:
        return "请输入 JD 内容"
    try:
        payload = {"jd_text": jd_text}
        if style:
            payload["style"] = style
        result = _post("/api/generate/self-intro", payload)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"生成失败: {e}"


def generate_project_intro(project_id: str, length: str) -> str:
    """生成项目介绍（STAR 格式）。"""
    if not project_id:
        return "请输入项目 ID"
    try:
        result = _post("/api/generate/project-intro", {
            "project_id": project_id,
            "length": length,
        })
        return result.get("intro", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        return f"生成失败: {e}"


def generate_cover_letter(company: str, position: str, jd_text: str, tone: str) -> str:
    """生成求职信。"""
    if not company or not position or not jd_text:
        return "请填写公司名称、岗位名称和 JD 内容"
    try:
        payload = {"company": company, "position": position, "jd_text": jd_text}
        if tone:
            payload["tone"] = tone
        result = _post("/api/generate/cover-letter", payload)
        return result.get("cover_letter", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        return f"生成失败: {e}"


STYLE_CHOICES = ["", "formal", "passionate", "concise"]
TONE_CHOICES = ["", "professional", "warm", "enthusiastic"]
LENGTH_CHOICES = ["short", "medium", "long"]


def build_generate_tab():
    """构建生成工具标签页。"""
    gr.Markdown("## ✨ 生成工具 — AI 辅助内容生成")
    gr.Markdown("基于您的简历和岗位信息，智能生成求职材料。")

    # ── 自我介绍 ──
    gr.Markdown("### 📝 生成自我介绍（自荐书）")
    with gr.Row():
        intro_jd = gr.Textbox(label="JD 全文", lines=4, scale=3)
        intro_style = gr.Dropdown(
            choices=STYLE_CHOICES, label="风格", value="", scale=1
        )
    intro_btn = gr.Button("🚀 生成自我介绍", variant="primary")
    intro_output = gr.Textbox(label="生成结果", lines=10)

    intro_btn.click(
        fn=generate_self_intro,
        inputs=[intro_jd, intro_style],
        outputs=intro_output,
    )

    # ── 项目介绍 ──
    gr.Markdown("---")
    gr.Markdown("### 🚀 生成项目介绍（STAR 格式）")
    with gr.Row():
        proj_id = gr.Textbox(label="项目 ID", scale=2)
        proj_length = gr.Radio(
            choices=LENGTH_CHOICES, label="篇幅", value="medium", scale=1
        )
    proj_btn = gr.Button("🚀 生成项目介绍", variant="primary")
    proj_output = gr.Textbox(label="生成结果", lines=8)

    proj_btn.click(
        fn=generate_project_intro,
        inputs=[proj_id, proj_length],
        outputs=proj_output,
    )

    # ── 求职信 ──
    gr.Markdown("---")
    gr.Markdown("### 📧 生成求职信")
    with gr.Row():
        cl_company = gr.Textbox(label="公司名称", scale=1)
        cl_position = gr.Textbox(label="岗位名称", scale=1)
        cl_tone = gr.Dropdown(
            choices=TONE_CHOICES, label="语气", value="", scale=1
        )
    cl_jd = gr.Textbox(label="JD 全文", lines=4)
    cl_btn = gr.Button("🚀 生成求职信", variant="primary")
    cl_output = gr.Textbox(label="生成结果", lines=10)

    cl_btn.click(
        fn=generate_cover_letter,
        inputs=[cl_company, cl_position, cl_jd, cl_tone],
        outputs=cl_output,
    )


# ═══════════════════════════════════════════════════════
# Tab 4: RAG 知识库 (Knowledge Base)
# ═══════════════════════════════════════════════════════


def rag_index_status() -> str:
    """获取 RAG 索引状态。"""
    try:
        data = _get("/api/rag/index/status")
        return (
            f"文档数: {data.get('document_count', 0)} | "
            f"索引类型: {data.get('index_type', 'N/A')} | "
            f"向量维度: {data.get('dimension', 'N/A')}"
        )
    except Exception as e:
        return f"获取状态失败: {e}"


def rag_import_documents(files: list[Any]) -> str:
    """上传文档到 RAG 知识库。"""
    if not files:
        return "请先选择文件（支持 PDF/DOCX/MD/TXT 格式）"
    try:
        with httpx.Client(timeout=120.0) as client:
            upload_files = []
            for f in files:
                upload_files.append(
                    ("files", (os.path.basename(f.name), open(f.name, "rb")))
                )
            resp = client.post(f"{BACKEND_URL}/api/rag/documents/import", files=upload_files)
            resp.raise_for_status()
            result = resp.json()
            return (
                f"导入成功! 处理了 {result.get('imported_files', 0)} 个文件, "
                f"创建了 {result.get('chunks_created', 0)} 个文本块, "
                f"知识库总计 {result.get('total_documents', 0)} 篇文档"
            )
    except Exception as e:
        return f"导入失败: {e}"


def rag_search(query: str, top_k: int = 5) -> str:
    """BM25 + 向量双路检索（不经过 LLM）。"""
    if not query:
        return "请输入搜索查询"
    try:
        result = _post("/api/rag/search", {"query": query, "top_k": top_k})
        if not result.get("results"):
            return "未找到相关文档"
        lines = [f"延迟: {result.get('latency_ms', 0)}ms\n"]
        for i, r in enumerate(result["results"], 1):
            lines.append(
                f"---\n### [{i}] 分数: {r.get('score', 0):.4f}\n{r.get('text', '')[:500]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


def rag_query(question: str, top_k: int = 5) -> str:
    """完整 RAG 查询：检索 → 重排序 → LLM 生成。"""
    if not question:
        return "请输入问题"
    try:
        result = _post("/api/rag/query", {
            "question": question,
            "top_k": top_k,
            "rerank_top_k": min(3, top_k),
        })
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        latency = result.get("latency_ms", 0)
        lines = [answer, f"\n\n---\n延迟: {latency}ms\n"]
        if sources:
            lines.append(f"\n### 参考来源 ({len(sources)}):")
            for i, s in enumerate(sources, 1):
                lines.append(f"\n**[{i}]** 分数: {s.get('score', 0):.4f}")
                lines.append(f"{s.get('content', '')[:300]}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


def rag_rebuild_index() -> str:
    """从 data/docs/ 目录重建索引。"""
    try:
        result = _post("/api/rag/index/rebuild", {})
        return (
            f"索引重建完成! 加载了 {result.get('documents_loaded', 0)} 个文档, "
            f"创建了 {result.get('chunks_created', 0)} 个文本块, "
            f"知识库总计 {result.get('total_documents', 0)} 篇文档"
        )
    except Exception as e:
        return f"重建失败: {e}"


def build_rag_tab():
    """构建 RAG 知识库标签页。"""
    gr.Markdown("## 📚 知识库 — RAG 检索增强生成")
    gr.Markdown("文档管理与智能问答：BM25+向量双路检索 → 重排序 → LLM 生成回答。")

    # ── 索引状态 ──
    gr.Markdown("### 📊 索引状态")
    with gr.Row():
        status_text = gr.Textbox(label="当前状态", interactive=False, scale=3)
        status_btn = gr.Button("🔄 刷新", scale=1)
        rebuild_btn = gr.Button("🔨 强制重建索引", variant="secondary", scale=1)

    status_btn.click(fn=rag_index_status, outputs=status_text)
    rebuild_btn.click(fn=rag_rebuild_index, outputs=status_text)

    gr.Markdown("---")

    # ── 文档导入 ──
    gr.Markdown("### 📤 导入文档")
    gr.Markdown("支持 PDF、DOCX、Markdown、TXT 格式。上传后自动解析、分块、嵌入并建立索引。")
    with gr.Row():
        upload_files = gr.File(
            label="选择文档",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".md", ".txt"],
            scale=3,
        )
        import_btn = gr.Button("🚀 开始导入", variant="primary", scale=1)
    import_output = gr.Textbox(label="导入结果", interactive=False)

    import_btn.click(fn=rag_import_documents, inputs=upload_files, outputs=import_output)

    gr.Markdown("---")

    # ── 搜索 ──
    gr.Markdown("### 🔍 向量+BM25 双路检索")
    gr.Markdown("直接检索知识库中的相关内容，不经过 LLM 生成。")
    with gr.Row():
        search_query = gr.Textbox(label="搜索关键词", placeholder="输入要搜索的内容...", scale=3)
        search_topk = gr.Slider(minimum=1, maximum=20, value=5, label="返回数量", scale=1)
    search_btn = gr.Button("🔍 搜索", variant="primary")
    search_output = gr.Markdown(label="搜索结果")

    search_btn.click(fn=rag_search, inputs=[search_query, search_topk], outputs=search_output)

    gr.Markdown("---")

    # ── 问答 ──
    gr.Markdown("### 💬 RAG 智能问答")
    gr.Markdown("基于知识库内容进行智能问答：检索 → 重排序 → LLM 生成回答（含来源标注）。")
    with gr.Row():
        qa_question = gr.Textbox(label="问题", placeholder="基于知识库内容提问...", lines=3, scale=3)
        qa_topk = gr.Slider(minimum=1, maximum=20, value=5, label="检索数量", scale=1)
    qa_btn = gr.Button("💬 提问", variant="primary")
    qa_output = gr.Markdown(label="回答")

    qa_btn.click(fn=rag_query, inputs=[qa_question, qa_topk], outputs=qa_output)


# ═══════════════════════════════════════════════════════
# Tab 5: 准备建议 (Prep)
# ═══════════════════════════════════════════════════════


def load_prep_plan(app_id: str) -> str:
    """加载指定投递记录的面试准备计划。"""
    if not app_id:
        return "请输入投递记录 ID"
    try:
        # 先获取投递记录详细信息
        data = _get(f"/api/applications/{app_id}")
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"加载失败: {e}"


def get_recommendations() -> str:
    """获取岗位推荐，返回格式化的 Markdown 展示。"""
    try:
        data = _post("/api/recommend", {})
        recs = data.get("recommendations", [])
        total = data.get("total_candidates", 0)
        if not recs:
            return "暂无推荐岗位，请先完善简历信息并添加 JD 数据。"

        lines = [f"从 {total} 个候选岗位中推荐 Top {len(recs)}：\n"]
        for i, r in enumerate(recs, 1):
            score = r.get("match_score", 0)
            score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(
                f"---\n### 🏆 推荐 #{i}：{r.get('company', '')} — {r.get('title', '')}"
            )
            lines.append(f"📍 {r.get('location', '未知')}  |  匹配度: **{score:.0%}**")
            lines.append(f"{score_bar}")
            if r.get("matched_skills"):
                lines.append(f"✅ 匹配技能: {'` · `'.join(r['matched_skills'][:5])}")
            if r.get("missing_skills"):
                lines.append(f"⚠️ 待补充: {'` · `'.join(r['missing_skills'][:5])}")
            lines.append(f"💡 {r.get('reason', '')}")
        return "\n".join(lines)
    except Exception as e:
        return f"获取推荐失败: {e}"


def build_prep_tab():
    """构建准备建议标签页。"""
    gr.Markdown("## 🎯 准备建议 — 面试准备与岗位匹配")
    gr.Markdown("查看投递记录详情和岗位推荐匹配分析。")

    gr.Markdown("### 🔍 投递记录详情")
    with gr.Row():
        prep_app_id = gr.Textbox(label="投递记录 ID", scale=3)
        prep_load_btn = gr.Button("📂 加载详情", variant="primary", scale=1)
    prep_output = gr.JSON(label="投递详情 & 准备计划")

    prep_load_btn.click(
        fn=load_prep_plan,
        inputs=prep_app_id,
        outputs=prep_output,
    )

    gr.Markdown("---")
    gr.Markdown("### 📊 岗位推荐（匹配度分析）")
    gr.Markdown("根据你的简历技能自动匹配岗位，展示匹配度、技能重叠和推荐理由。")
    rec_btn = gr.Button("🔄 获取岗位推荐", variant="secondary")
    rec_output = gr.Markdown(label="推荐结果")

    rec_btn.click(fn=get_recommendations, outputs=rec_output)


# ═══════════════════════════════════════════════════════
# Tab 5: 配置 (Settings)
# ═══════════════════════════════════════════════════════


def save_settings(
    wechat_appid: str,
    wechat_appsecret: str,
    daily_log_enabled: bool,
    daily_log_time: str,
    schedule_review_time: str,
) -> str:
    """保存配置（写入 .env 文件）。"""
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        # 读取现有 .env
        lines = []
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        # 更新或追加
        updates = {
            "WECHAT_APPID": wechat_appid,
            "WECHAT_APPSECRET": wechat_appsecret,
            "DAILY_LOG_ENABLED": str(daily_log_enabled).lower(),
            "DAILY_LOG_TIME": daily_log_time,
            "SCHEDULE_REVIEW_TIME": schedule_review_time,
        }

        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return f"✅ 配置已保存到 {env_path}"
    except Exception as e:
        return f"保存配置失败: {e}"


def check_health() -> str:
    """检查后端健康状态。"""
    try:
        result = _get("/health")
        return f"✅ 后端运行正常 — 状态: {result.get('status', 'ok')}, 版本: {result.get('version', '?')}"
    except Exception as e:
        return f"❌ 后端连接失败: {e}"


def build_settings_tab():
    """构建配置标签页。"""
    gr.Markdown("## ⚙️ 配置 — 系统设置")
    gr.Markdown("配置微信、日志和定时任务参数。")

    # 健康检查
    health_btn = gr.Button("🩺 检查后端连接", variant="secondary")
    health_output = gr.Textbox(label="连接状态", interactive=False)
    health_btn.click(fn=check_health, outputs=health_output)

    gr.Markdown("---")
    gr.Markdown("### 💬 微信配置")

    with gr.Row():
        wechat_appid = gr.Textbox(label="WECHAT_APPID", placeholder="微信应用 ID")
        wechat_appsecret = gr.Textbox(
            label="WECHAT_APPSECRET",
            placeholder="微信应用密钥",
            type="password",
        )

    gr.Markdown("---")
    gr.Markdown("### ⏰ 定时任务配置")

    with gr.Row():
        daily_log_enabled = gr.Checkbox(label="启用每日日志", value=True)
        daily_log_time = gr.Textbox(
            label="每日日志时间", value="22:00", placeholder="HH:MM"
        )
        schedule_review_time = gr.Textbox(
            label="定时回顾时间", value="09:30", placeholder="HH:MM"
        )

    save_btn = gr.Button("💾 保存配置", variant="primary")
    save_output = gr.Textbox(label="保存结果", interactive=False)

    save_btn.click(
        fn=save_settings,
        inputs=[
            wechat_appid,
            wechat_appsecret,
            daily_log_enabled,
            daily_log_time,
            schedule_review_time,
        ],
        outputs=save_output,
    )


# ═══════════════════════════════════════════════════════
# 主应用
# ═══════════════════════════════════════════════════════

CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; }
button[role="tab"] { pointer-events: auto !important; position: relative; z-index: 100 !important; }
div[role="tablist"] { position: relative; z-index: 50 !important; }
"""

with gr.Blocks(
    title="智能招聘助手 v2.0",
) as app:
    gr.Markdown(
        "# 🏗 智能招聘助手 v2.0 — 求职 Agent 智能助手"
    )
    gr.Markdown(
        "简历管理 · 投递跟踪 · AI 生成 · 知识库问答 · 面试准备 · 岗位推荐"
    )

    with gr.Tabs():
        with gr.TabItem("📄 资料库"):
            build_profile_tab()
        with gr.TabItem("📋 投递管理"):
            build_applications_tab()
        with gr.TabItem("✨ 生成工具"):
            build_generate_tab()
        with gr.TabItem("📚 知识库"):
            build_rag_tab()
        with gr.TabItem("🎯 准备建议"):
            build_prep_tab()
        with gr.TabItem("⚙️ 配置"):
            build_settings_tab()

    gr.Markdown("---")
    gr.Markdown(
        "<center>智能招聘助手 v2.0 | Powered by FastAPI + Gradio + DeepSeek</center>"
    )

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    # 允许在已有事件循环中运行 Gradio
    nest_asyncio.apply()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=CSS) # type: ignore
