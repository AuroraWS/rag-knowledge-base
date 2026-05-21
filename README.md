# 🏗 智能招聘助手 v2.0 — 求职 Agent 智能助手

> FastAPI + FAISS + bge-reranker + DeepSeek + LangGraph + Gradio

基于 **个人简历知识库** 的智能求职助手，支持简历管理、投递跟踪、AI 内容生成、面试准备和岗位推荐。

---

## 📋 项目概览

智能招聘助手 v2.0 是一个面向求职者的一站式 Agent 智能助手系统：

- 📄 **简历管理** — 上传、解析、存储个人简历（基本信息、教育、工作、项目、证书）
- 📋 **投递跟踪** — 记录投递进度，管理投递状态，自动提醒跟进
- ✨ **AI 生成** — 根据 JD 和简历生成自我介绍、项目介绍（STAR 格式）、求职信
- 🎯 **面试准备** — 岗位匹配度分析，结构化面试准备建议
- ⚙️ **配置灵活** — 微信机器人、定时任务调度

---

## 🛠 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek API |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量库 | FAISS (HNSW-PQ) |
| 稀疏检索 | BM25 |
| Agent 编排 | LangGraph |
| 后端 | FastAPI (Python 3.11) |
| 前端 | Gradio (多页面) |
| 存储 | SQLite + JSON 文件 |
| 部署 | Docker / docker-compose |

---

## 📁 目录结构

```
rag-knowledge-base/
├── app/
│   ├── main.py                # FastAPI 入口
│   ├── config.py              # 配置管理 (.env)
│   ├── api/                   # API 路由（5 个模块）
│   │   ├── profile.py         #   简历信息管理
│   │   ├── generate.py        #   LLM 内容生成
│   │   ├── applications.py    #   投递记录 CRUD
│   │   ├── command.py         #   自然语言命令
│   │   └── recommend.py       #   岗位推荐匹配
│   ├── models/                # Pydantic 数据模型
│   ├── rag/                   # RAG 核心（检索+生成）
│   ├── agent/                 # 多 Agent 工作流
│   ├── services/              # 业务逻辑层
│   ├── storage/               # 持久化存储层
│   └── data/                  # 文档加载+示例数据
├── frontend/
│   └── app.py                 # Gradio 多页面前端
├── tests/                     # 测试
├── scripts/                   # 工具脚本
├── data/                      # 运行时数据目录
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量模板
├── Dockerfile                 # Docker 部署
└── README.md                  # 本文件
```

---

## 🚀 快速启动

### 方式一：venv + pip（推荐）

```bash
# 克隆项目
cd rag-knowledge-base

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 下载 embedding 模型（首次）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 启动后端 API 服务
uvicorn app.main:app --reload

# （新终端）启动 Gradio 前端
python frontend/app.py
```

### 方式二：Docker

```bash
# 构建镜像
docker build -t smart-hire:2.0 .

# 运行（需要先配置 .env）
docker run -p 8000:8000 -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  smart-hire:2.0
```

---

## 📡 API 概览

### 简历信息管理 (`/api/profile`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/profile` | 获取完整简历数据 |
| PUT | `/api/profile/personal-info` | 更新个人基本信息 |
| POST | `/api/profile/education` | 添加教育经历 |
| POST | `/api/profile/work-experience` | 添加工作经历 |
| POST | `/api/profile/project` | 添加项目经历 |
| POST | `/api/profile/certificate` | 添加证书/资质 |
| POST | `/api/profile/upload` | 上传简历文件 |

### LLM 内容生成 (`/api/generate`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/generate/self-intro` | 生成自我介绍（三段式） |
| POST | `/api/generate/project-intro` | 生成项目介绍（STAR） |
| POST | `/api/generate/cover-letter` | 生成求职信 |

### 投递管理 (`/api/applications`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/applications` | 获取投递列表（支持状态筛选） |
| POST | `/api/applications` | 添加投递记录 |
| GET | `/api/applications/stats` | 获取投递统计 |
| GET | `/api/applications/{id}` | 获取单条记录 |
| PUT | `/api/applications/{id}` | 更新投递记录 |
| PUT | `/api/applications/{id}/status` | 更新投递状态 |
| DELETE | `/api/applications/{id}` | 删除投递记录 |

### 自然语言命令 (`/api/command`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/command` | 解析并执行自然语言命令 |

### 岗位推荐 (`/api/recommend`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/recommend` | 获取岗位推荐（Top 5） |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |

---

## ✨ 功能特性

### 1. 📄 资料库（Profile）
简历信息集中管理，支持个人基本信息、教育经历、工作经历、项目经历、证书/资质的增删改查。支持上传 PDF/Word 简历文件进行自动化解析（存根阶段）。

### 2. 📋 投递管理（Applications）
投递记录全生命周期跟踪。支持按状态筛选、添加新投递、更新状态（自动追加时间线）、待跟进提醒。

### 3. ✨ 生成工具（Generate）
基于 DeepSeek API 和简历数据，智能生成：
- **自我介绍** — 短版/完整版/英文版，可指定风格
- **项目介绍** — STAR 格式，支持三种篇幅
- **求职信** — 按公司和岗位定制，可选语气

### 4. 🎯 准备建议（Prep）
投递记录详情查看 + 岗位推荐匹配分析。基于简历技能关键词与 JD 数据集进行匹配度计算，返回 Top 5 推荐岗位及匹配详情。

### 5. ⚙️ 配置（Settings）
微信机器人配置（AppID / AppSecret），定时任务调度（每日日志、定时回顾）。

---

## 🔧 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-your-api-key-here` |
| `DEEPSEEK_BASE_URL` | API 基础地址 | `https://api.deepseek.com` |
| `EMBEDDING_MODEL` | Embedding 模型名称 | `BAAI/bge-small-zh-v1.5` |
| `RERANKER_MODEL` | Reranker 模型名称 | `BAAI/bge-reranker-v2-m3` |
| `HOST` | 后端监听地址 | `0.0.0.0` |
| `PORT` | 后端监听端口 | `8000` |
| `WECHAT_APPID` | 微信机器人 AppID | — |
| `WECHAT_APPSECRET` | 微信机器人 AppSecret | — |
| `DAILY_LOG_ENABLED` | 启用每日日志 | `true` |
| `DAILY_LOG_TIME` | 每日日志生成时间 | `22:00` |
| `SCHEDULE_REVIEW_TIME` | 定时回顾时间 | `09:30` |

详见 `.env.example`。

---

## 🧪 测试

```bash
pytest tests/ -v
```

---

## 📄 License

MIT
