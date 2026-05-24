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
TIMEOUT = 5.0


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
    except Exception as e:
        return f"上传失败: {e}"


def build_profile_tab() -> gr.Blocks:
    """构建资料库标签页。"""
    with gr.Blocks() as tab:
        gr.Markdown("## 📄 资料库 — 个人简历信息")
        gr.Markdown("查看和更新您的个人基本信息、教育经历、工作经历、项目经历和证书。")

        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新简历", variant="primary")
            upload_btn = gr.UploadButton("📤 上传简历文件", file_types=[".pdf", ".docx", ".doc", ".png", ".jpg"])

        profile_output = gr.JSON(label="简历数据 (JSON)")

        refresh_btn.click(fn=load_profile, outputs=profile_output)
        upload_btn.upload(fn=upload_doc, inputs=upload_btn, outputs=profile_output)

        # 自动加载一次
        gr.on(triggers=tab.load, fn=load_profile, outputs=profile_output)
    return tab


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


def build_applications_tab() -> gr.Blocks:
    """构建投递管理标签页。"""
    with gr.Blocks() as tab:
        gr.Markdown("## 📋 投递管理 — 投递记录跟踪")
        gr.Markdown("查看所有投递记录，添加新记录，更新状态。")

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
            row_count=(10, "dynamic"), # type: ignore
            interactive=False,
        )

        refresh_btn.click(fn=list_apps, inputs=status_filter, outputs=apps_table)
        status_filter.change(fn=list_apps, inputs=status_filter, outputs=apps_table)
        gr.on(triggers=tab.load, fn=list_apps, inputs=status_filter, outputs=apps_table)

        gr.Markdown("---")
        gr.Markdown("### ➕ 添加投递记录")

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

    return tab


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


def build_generate_tab() -> gr.Blocks:
    """构建生成工具标签页。"""
    with gr.Blocks() as tab:
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

    return tab


# ═══════════════════════════════════════════════════════
# Tab 4: 准备建议 (Prep)
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
    """获取岗位推荐。"""
    try:
        data = _post("/api/recommend", {})
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"获取推荐失败: {e}"


def build_prep_tab() -> gr.Blocks:
    """构建准备建议标签页。"""
    with gr.Blocks() as tab:
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
        rec_btn = gr.Button("🔄 获取岗位推荐", variant="secondary")
        rec_output = gr.JSON(label="推荐结果")

        rec_btn.click(fn=get_recommendations, outputs=rec_output)

    return tab


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


def build_settings_tab() -> gr.Blocks:
    """构建配置标签页。"""
    with gr.Blocks() as tab:
        gr.Markdown("## ⚙️ 配置 — 系统设置")
        gr.Markdown("配置微信、日志和定时任务参数。")

        # 健康检查
        health_btn = gr.Button("🩺 检查后端连接", variant="secondary")
        health_output = gr.Textbox(label="连接状态", interactive=False)
        health_btn.click(fn=check_health, outputs=health_output)
        gr.on(triggers=tab.load, fn=check_health, outputs=health_output)

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

    return tab


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
        "简历管理 · 投递跟踪 · AI 生成 · 面试准备 · 岗位推荐"
    )

    with gr.Tabs():
        with gr.TabItem("📄 资料库"):
            build_profile_tab()
        with gr.TabItem("📋 投递管理"):
            build_applications_tab()
        with gr.TabItem("✨ 生成工具"):
            build_generate_tab()
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
