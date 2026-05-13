# RAG 企业知识库智能问答系统

> FastAPI + FAISS + bge-reranker + DeepSeek + Multi-Agent

基于 RAG（检索增强生成）的企业级智能知识库，支持文档上传、语义检索、LLM 问答。

---

## 🚀 快速启动

### 方式一：Conda（推荐）

```bash
# 创建环境
conda env create -f environment.yml
conda activate rag-knowledge-base

# 配置
cp .env.example .env
# 编辑 .env 填上 DEEPSEEK_API_KEY

# 下载模型（首次）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 启动
uvicorn app.main:app --reload
```

### 方式二：venv + pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env 填上 DEEPSEEK_API_KEY

uvicorn app.main:app --reload
```

---

## 📁 项目结构

```
rag-knowledge-base/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── models/schema.py     # Pydantic 数据模型
│   ├── rag/                 # RAG 核心（检索+生成）
│   ├── agent/               # 多 Agent 协作
│   └── data/                # 文档加载+切分+索引
├── tests/
├── scripts/
├── knowledge_base/          # 原始文档
├── faiss_index/             # FAISS 向量索引
├── environment.yml          # Conda 环境
├── requirements.txt         # Pip 依赖（备选）
└── Dockerfile               # Docker 部署
```

---

## 🛠 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek API |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量库 | FAISS (HNSW-PQ) |
| 稀疏检索 | BM25 |
| Agent | LangGraph / AutoGen |
| 后端 | FastAPI |
| Demo | Gradio |
| 部署 | Docker + docker-compose |

---

## 📡 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/rag/query` | RAG 问答（开发中） |

---

## 📄 License

MIT
