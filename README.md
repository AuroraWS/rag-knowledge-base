# 🧠 智能招聘助手 (SmartHire)

> AI 驱动的智能招聘平台 — RAG + Multi-Agent + 知识图谱

## 🎯 项目定位

一个即可 ToB（HR智能匹配候选人）又可 ToC（求职者简历优化+职位推荐）的智能招聘助手。

## 🏗️ 技术架构

```
LLM: DeepSeek API
Embedding: BAAI/bge-small-zh-v1.5 (本地)
Reranker: bge-reranker-v2-m3 (本地)
向量库: Qdrant
Agent编排: LangGraph StateGraph
后端: FastAPI + Pydantic
前端: Gradio
部署: Docker + docker-compose
CI/CD: GitHub Actions
```

## 🚀 快速开始

```bash
# 1. 克隆
git clone <repo-url> && cd smart-hire

# 2. 环境
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. 配置
cp .env.example .env  # 填你的 DeepSeek API Key

# 4. 数据准备
python scripts/generate_data.py  # 生成模拟简历+JD

# 5. 构建索引
python scripts/build_index.py

# 6. 启动
uvicorn app.main:app --reload
# 打开 http://localhost:8000
```

## 📂 目录结构

```
smart-hire/
├── app/              # FastAPI 应用
│   ├── main.py       # 入口
│   ├── agents/       # 多Agent协作
│   ├── rag/          # RAG Pipeline
│   ├── knowledge/    # 知识图谱
│   └── models/       # Pydantic 模型
├── data/             # 数据集 (JD + 简历)
├── scripts/          # 建索引/生成数据脚本
├── tests/            # 测试
├── docker/           # Docker 配置
└── docs/             # 文档
```

## 🧪 核心功能

| 功能 | 状态 |
|------|------|
| ToC: 简历上传 + 词条优化 | 🚧 开发中 |
| ToC: 职位智能匹配 | 🚧 开发中 |
| ToC: 薪资评估 | 🚧 开发中 |
| ToC: 职业发展方案 | 🚧 开发中 |
| ToB: JD输入 + 候选人匹配 | 🚧 开发中 |

## 📄 License

MIT
