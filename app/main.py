"""FastAPI 入口 — 智能招聘助手 v2.0

注册所有 API 路由，配置 CORS，启动时初始化数据目录和存储。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schema import HealthResponse
from app.api import (
    profile_router,
    generate_router,
    applications_router,
    command_router,
    recommend_router,
    rag_router,
)

app = FastAPI(
    title="智能招聘助手 v2.0",
    description="求职 Agent 智能助手",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由注册 ──────────────────────────────────────

app.include_router(profile_router)
app.include_router(generate_router)
app.include_router(applications_router)
app.include_router(command_router)
app.include_router(recommend_router)
app.include_router(rag_router)


# ── 健康检查 ──────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(version="2.0.0")


# ── WeChat Bot Webhook ──────────────────────────


from app.gateway.wechat import wechat_adapter
from app.gateway.router import message_router
from fastapi import Request, Response


@app.get("/api/wechat/webhook")
async def wechat_verify(signature: str, timestamp: str, nonce: str, echostr: str):
    """微信服务器首次接入验证（GET请求）。"""
    result = wechat_adapter.verify_signature(signature, timestamp, nonce, echostr)
    return Response(content=result if isinstance(result, str) else str(result))


@app.post("/api/wechat/webhook")
async def wechat_receive(request: Request):
    """接收微信消息推送（POST请求）。"""
    body = await request.body()
    xml_data = body.decode("utf-8")

    msg = wechat_adapter.parse_message(xml_data)
    reply_text = await message_router.route(msg)

    reply_xml = wechat_adapter.build_reply(
        to_user=msg.from_user,
        from_user=msg.raw.get("to_user", "gh_default"),
        content=reply_text,
    )
    return Response(content=reply_xml, media_type="application/xml")


# ── 启动事件 ──────────────────────────────────────


@app.on_event("startup")
async def startup():
    """在应用启动时初始化数据目录和存储层。"""
    import os
    from pathlib import Path

    # 确保所有数据目录存在
    data_dirs = [
        settings.profile_dir,
        settings.docs_dir,
        settings.memory_dir,
        settings.log_dir,
        os.path.dirname(settings.sqlite_path),
        settings.faiss_index_dir,
    ]
    for d in data_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 初始化存储（触发懒加载，创建默认数据文件）
    from app.storage.profile_store import profile_store
    from app.storage.application_store import application_store
    from app.storage.memory_store import memory_store

    # 触碰每个 store 使其完成懒初始化
    _ = profile_store.get_all()
    _ = application_store.list()
    _ = memory_store.all()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
