# 🏗 智能招聘助手 v2.0 — 求职 Agent 智能助手

> FastAPI + FAISS HNSW-PQ + BM25 + bge-reranker + DeepSeek + LangGraph + Gradio

---

## 背景

求职过程中信息分散、重复劳动多——每次投简历都要重新翻找学位证编号、重写自我评价、追踪十几家公司的投递进度。市面上没有工具能把这整条链路串起来。

**我做了个"装在自己电脑上的求职副驾"**，核心解决四个痛点：

| 痛点 | 方案 |
|:-----|:-----|
| 每次投递都要翻箱倒柜找资料 | 存一次，AI 自动填 |
| 自我评价/项目介绍每次重写 | RAG + LLM 自动生成 |
| 投了十几家记不清状态 | 自然语言一条指令搞定 CRUD |
| 不知道什么岗位适合自己 | 技能向量匹配 + 智能推荐 |

**一句话定位**：不是招聘平台，不是 HR 工具，是求职者的 AI Agent 私人助理。

---

## 🛠 技术栈

| 组件 | 选型 | 说明 |
|:-----|:-----|:-----|
| LLM | DeepSeek API (v4-pro/flash) | 生成/意图识别/文档结构化 |
| Embedding | BAAI/bge-small-zh-v1.5 | L2 归一化，内积即余弦相似度 |
| Reranker | BAAI/bge-reranker-v2-m3 | CrossEncoder 精排 |
| 向量库 | FAISS (Flat/HNSW/IVF/HNSW_PQ) | 4 种索引可切换，train 失败自动回退 |
| 稀疏检索 | BM25 (rank_bm25) | 中文按字切分 |
| 融合 | RRF (k=60) | 双路排名融合 |
| Agent 编排 | LangGraph StateGraph | 5 节点，不可用时自动回退顺序执行 |
| 后端 | FastAPI (Python 3.11) | 异步，38 个 API 端点 |
| 前端 | Gradio 6 Tab | 自然语言指令 + 格式化推荐卡片 |
| 存储 | SQLite + JSON 文件 + FAISS 索引 | 纯本地，零依赖 |
| 部署 | Docker + docker-compose | 多阶段构建 + healthcheck |

---

## 🚀 快速启动

```bash
# 创建环境
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 配置
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
# 国内用户设置 HF_ENDPOINT=https://hf-mirror.com

# 下载模型（首次）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 启动后端
uvicorn app.main:app --reload

# 启动前端（新终端）
python frontend/app.py
```

访问：后端 `http://localhost:8000/docs` | 前端 `http://localhost:7860`

---

## 📁 目录结构

```
rag-knowledge-base/
├── app/
│   ├── main.py                 # FastAPI 入口 (38 routes)
│   ├── config.py               # 配置管理 (.env)
│   ├── api/                    # API 路由 (7 个模块)
│   │   ├── profile.py          #   简历管理 + AI 提取确认
│   │   ├── rag.py              #   RAG 知识库 (7 端点)
│   │   ├── generate.py         #   LLM 内容生成
│   │   ├── applications.py     #   投递记录 CRUD
│   │   ├── command.py          #   自然语言命令解析
│   │   └── recommend.py        #   岗位推荐匹配
│   ├── models/                 # Pydantic 数据模型 (schema.py)
│   ├── rag/                    # RAG 核心引擎
│   │   ├── embeddings.py       #   BGE Embedding (HF 镜像 + Pooling 自动修补)
│   │   ├── retriever.py        #   BM25 + FAISS 双路检索 + 持久化
│   │   ├── reranker.py         #   CrossEncoder 精排
│   │   ├── generator.py        #   DeepSeek LLM (生成 + 流式 + JSON 安全解析)
│   │   └── pipeline.py         #   RAG 完整流水线 + 简历-JD 匹配
│   ├── agent/                  # 多 Agent 编排
│   │   ├── rag_agent.py        #   查询改写 → 检索规划 → 合成 → 护栏
│   │   ├── workflow.py         #   LangGraph 投递/面试工作流
│   │   └── scheduler.py        #   APScheduler 定时任务
│   ├── data/                   # 文档处理
│   │   ├── loader.py           #   文档加载器 (PDF/DOCX/MD/TXT)
│   │   ├── pdf_parser.py       #   PDF 解析器 (pymupdf + DeepSeek 管线)
│   │   └── chunker.py          #   多策略切分 (合同/法规/FAQ/简历/通用)
│   ├── services/               # 业务逻辑层
│   │   ├── extraction_service.py  # 字段提取 (规则 + LLM)
│   │   ├── generation_service.py  # 内容生成
│   │   ├── preparation_service.py # 面试准备
│   │   └── tracking_service.py    # 投递追踪
│   ├── storage/                # 持久化存储层
│   │   ├── profile_store.py    #   简历 JSON 存储
│   │   ├── application_store.py#   投递 SQLite 存储
│   │   └── memory_store.py     #   字段记忆存储
│   └── gateway/                # 微信网关
│       ├── wechat.py           #   消息收发 + 签名验证
│       ├── router.py           #   消息路由
│       └── push.py             #   主动推送管理
├── frontend/
│   └── app.py                  # Gradio 6 页面前端
├── tests/                      # 45 个测试 (37 unit + 3 integration + 3 regression + 8 新增)
├── data/                       # 运行时数据 (profile/faiss_index/logs/docs)
├── requirements.txt
├── Dockerfile                  # 多阶段构建
├── docker-compose.yml          # backend + frontend
└── .env.example
```

---

## 📡 API 概览 (38 端点)

### 简历管理 (`/api/profile`)

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| GET | `/api/profile` | 获取完整简历 |
| PUT | `/api/profile/personal-info` | 更新个人信息 |
| POST | `/api/profile/education` | 添加教育经历 |
| POST | `/api/profile/work-experience` | 添加工作经历 |
| POST | `/api/profile/project` | 添加项目经历 |
| POST | `/api/profile/certificate` | 添加证书 |
| POST | `/api/profile/upload/extract` | 🆕 上传文件 → AI 提取预览 |
| POST | `/api/profile/upload/confirm` | 🆕 确认/修正后入库 |
| POST | `/api/profile/upload` | 上传 → 提取 → 直接入库 |

### RAG 知识库 (`/api/rag`)

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| POST | `/api/rag/documents/import` | 导入文档 (支持 5 种切分策略) |
| POST | `/api/rag/search` | BM25 + FAISS 双路检索 |
| POST | `/api/rag/query` | 完整 RAG: 检索 → 重排序 → LLM 生成 |
| POST | `/api/rag/query/stream` | 🆕 流式 SSE RAG 查询 |
| GET | `/api/rag/index/status` | 索引统计 |
| POST | `/api/rag/index/rebuild` | 强制重建索引 |
| POST | `/api/rag/agent/query` | 🆕 多 Agent RAG 查询 |

### 其他模块

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| POST | `/api/command` | 自然语言指令解析执行 |
| POST | `/api/generate/self-intro` | 生成自我介绍 |
| POST | `/api/generate/project-intro` | 生成项目介绍 (STAR) |
| POST | `/api/generate/cover-letter` | 生成求职信 |
| GET/POST | `/api/applications` | 投递记录 CRUD |
| POST | `/api/recommend` | 岗位推荐 (Top 5) |
| GET/POST | `/api/wechat/webhook` | 微信 Bot 接入 |
| GET | `/health` | 健康检查 |

---

## 🔧 环境变量

| 变量 | 说明 | 默认值 |
|:-----|:-----|:-------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | (必填) |
| `DEEPSEEK_BASE_URL` | API 基础地址 | `https://api.deepseek.com` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-small-zh-v1.5` |
| `RERANKER_MODEL` | Reranker 模型 | `BAAI/bge-reranker-v2-m3` |
| `HF_ENDPOINT` | 🆕 HuggingFace 镜像 | (空=官方，国内设 `https://hf-mirror.com`) |
| `LOCAL_FILES_ONLY` | 🆕 仅用本地模型缓存 | `false` |
| `FAISS_INDEX_TYPE` | 🆕 索引类型 (Flat/HNSW/IVF/HNSW_PQ) | `Flat` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 🆕 文档切分参数 | `500` / `50` |
| `HOST` / `PORT` | 服务地址 | `0.0.0.0` / `8000` |
| `WECHAT_APPID` / `WECHAT_APPSECRET` | 微信 Bot 配置 | (可选) |
| `DAILY_LOG_ENABLED` | 启用每日日志 | `true` |
| `DAILY_LOG_TIME` | 日志生成时间 | `22:00` |
| `SCHEDULE_REVIEW_TIME` | 定时回顾时间 | `09:30` |

---

## 🧪 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行含集成测试
pytest tests/ -v --runintegration

# 运行含慢速回归测试
pytest tests/ -v --runslow
```

测试结构：`tests/unit/` (37) + `tests/integration/` (3) + `tests/regression/` (3) + 新增 chunker 策略测试 (8) = **51 tests**

---

## 🐛 已修复的关键 Bug

### 本次迭代 (2026-05-24)

| Bug | 根因 | 修复 | 文件 |
|:----|:-----|:-----|:-----|
| Tab 切换完全无响应 | `gr.Blocks()` 嵌套在 `gr.TabItem` 内——每个 tab 被包成独立 Gradio 应用，事件被拦截 | 去掉 6 个 `build_*_tab()` 中的 `with gr.Blocks() as tab:` 包裹 | `frontend/app.py` |
| 上传简历返回 `saved: {}` | LLM 返回的 JSON 有时包裹在 markdown 代码块 (```json...```) 中，`json.loads` 解析失败 → 静默降级为空 | 新增 `_safe_json_parse()` 自动剥离代码块 | `app/rag/generator.py` |
| 前端 LLM 操作超时 | `TIMEOUT=5s` 对 AI 生成操作太短（LLM 需 3-15s） | 改为 `connect=5s, read=60s`——连接快速失败，读取等 AI | `frontend/app.py` |
| Embedding 模型加载失败 | `cache_folder` 从模型名错误推导 + `local_files_only=True` 硬编码 + huggingface.co 被墙 | 删除错误 cache_folder、添加 `HF_ENDPOINT` 镜像配置、`local_files_only` 改为可配置、自动修补 Pooling 配置兼容 sentence_transformers 5.x | `app/rag/embeddings.py` `app/rag/reranker.py` `app/config.py` |
| 端口 7860 / 8000 被占 | Windows Ctrl+C 后子进程残留 | 添加端口检查命令到文档 | — |

### 之前修复

| Bug | 说明 |
|:----|:-----|
| FAISS 索引重启丢失 | `build_index()` 后未持久化 → 新增自动 `_persist()` + 启动时 `try_load_from_disk()` |
| BM25 单文档 IDF 为 0 | 1 篇文档时全部 IDF=0 → 测试用 3 篇文档规避 |
| test mock SentenceTransformer 失败 | 局部导入导致 patch 路径错误 → 改在 `sentence_transformers` 源处 patch |
| Git HTTPS push 被 GFW 阻断 | github.com:443 不可达 → 切换 SSH + `ssh.github.com:443` |

---

## ✨ 核心功能

### RAG 检索链路
```
用户问题 → [Query Rewrite] → BM25+FAISS 双路检索 → RRF(k=60)融合
→ CrossEncoder 精排 → DeepSeek LLM 生成 → [Guardrails 安全+幻觉检测]
```

### 多 Agent 编排
```
rewrite_query → plan_retrieval → execute_search → synthesize → guardrails
```

### 多策略文档切分
| 类型 | 策略 |
|:-----|:-----|
| 合同 | 按条款编号切 (第X条) |
| 法规 | 双层: 章→条，附上层上下文 |
| FAQ | 按问答对切分 |
| 简历 | 按模块切分 (教育/工作/项目) |
| 通用 | RecursiveCharacterTextSplitter fallback |

---

## 📄 License

MIT
