# ============================================
# Dockerfile — 智能招聘助手 v2.0 (求职 Agent)
# ============================================

# ── 阶段1：构建 ──
FROM python:3.11-slim AS builder

WORKDIR /app

# 先装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 阶段2：运行 ──
FROM python:3.11-slim

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY .env ./

# 暴露端口（后端: 8000, 前端 Gradio: 7860）
EXPOSE 8000
EXPOSE 7860

# 默认启动后端服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
