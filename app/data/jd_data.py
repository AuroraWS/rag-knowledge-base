"""示例 JD 数据集——用于岗位推荐匹配引擎。

每个 JD 是 Pydantic BaseModel，包含：
- company, title, location: 基本信息
- raw_text: JD 全文（用于关键词匹配）
- requirements: 结构化的任职要求列表
"""

from pydantic import BaseModel, Field


class JDItem(BaseModel):
    company: str
    title: str
    location: str
    raw_text: str = ""
    requirements: list[str] = Field(default_factory=list)


jds: list[JDItem] = [
    JDItem(
        company="字节跳动",
        title="AI 应用开发工程师",
        location="北京",
        raw_text=(
            "负责大模型应用落地，包括 RAG 检索增强生成系统、Agent 智能体开发。"
            "要求精通 Python，熟悉 LangChain 或 LlamaIndex 框架，"
            "有 FAISS、向量数据库使用经验，了解 BGE 等 Embedding 模型。"
            "硕士及以上学历，计算机相关专业，有 NLP 项目经验优先。"
        ),
        requirements=["Python", "LangChain", "FAISS", "RAG", "NLP", "硕士"],
    ),
    JDItem(
        company="阿里巴巴",
        title="大模型算法工程师",
        location="杭州",
        raw_text=(
            "负责大语言模型训练和微调，包括 SFT、RLHF 等对齐技术。"
            "要求精通 PyTorch 或 TensorFlow，熟悉 Transformer 架构，"
            "有深度学习模型部署经验，了解 Docker、Kubernetes。"
            "计算机科学硕士及以上，有 ACL/EMNLP 等顶会论文优先。"
        ),
        requirements=["PyTorch", "TensorFlow", "Transformer", "Docker", "深度学习", "硕士"],
    ),
    JDItem(
        company="腾讯",
        title="AI 产品经理（大模型方向）",
        location="深圳",
        raw_text=(
            "负责大模型相关产品的规划和设计，包括对话系统、知识库问答等场景。"
            "要求理解大模型能力和边界，能独立完成产品需求文档，"
            "有技术背景，了解基本的机器学习概念。"
            "本科及以上学历，3 年以上 AI 或 SaaS 产品经验。"
        ),
        requirements=["产品设计", "机器学习", "大模型", "本科"],
    ),
    JDItem(
        company="华为",
        title="云计算 AI 开发工程师",
        location="上海",
        raw_text=(
            "负责华为云 AI 平台开发和维护，包括模型训练平台、推理服务。"
            "要求精通 Python 或 Java，有云服务开发经验，"
            "熟悉 Docker、Kubernetes，了解 CI/CD 流程。"
            "本科及以上学历，3 年以上后端开发或 AI 工程化经验。"
        ),
        requirements=["Python", "Java", "Docker", "Kubernetes", "云服务", "本科"],
    ),
    JDItem(
        company="小米",
        title="NLP 算法研究员",
        location="北京",
        raw_text=(
            "负责自然语言处理算法研发，包括文本分类、实体识别、问答系统等。"
            "要求精通 Python 和 PyTorch，熟悉 BERT、GPT 等预训练模型，"
            "有知识图谱、RAG 检索增强生成经验优先。"
            "硕士及以上学历，ACL/EMNLP/NAACL 等顶会论文发表经验。"
        ),
        requirements=["Python", "PyTorch", "BERT", "NLP", "RAG", "知识图谱", "硕士"],
    ),
    JDItem(
        company="美团",
        title="搜索推荐算法工程师",
        location="北京",
        raw_text=(
            "负责美团搜索和推荐系统的算法优化，包括召回、粗排、精排等环节。"
            "要求精通 Python/C++，熟悉 Elasticsearch、FAISS 等检索引擎，"
            "了解 BM25、向量检索、深度学习排序模型。"
            "本科及以上学历，有搜索/推荐/广告相关经验优先。"
        ),
        requirements=["Python", "C++", "Elasticsearch", "FAISS", "BM25", "向量检索", "本科"],
    ),
    JDItem(
        company="比亚迪",
        title="自动驾驶感知算法工程师",
        location="深圳",
        raw_text=(
            "负责自动驾驶感知算法研发，包括目标检测、语义分割、多传感器融合。"
            "要求精通 Python 和 PyTorch，熟悉计算机视觉基础算法，"
            "有 TensorRT、ONNX 等模型部署优化经验。"
            "硕士及以上学历，有顶会论文或自动驾驶项目经验优先。"
        ),
        requirements=["Python", "PyTorch", "计算机视觉", "深度学习", "TensorRT", "硕士"],
    ),
    JDItem(
        company="中国商飞",
        title="人工智能场景开发工程师",
        location="上海浦东",
        raw_text=(
            "负责航空领域 AI 场景落地，包括大模型微调、RAG 知识库构建、Agent 工作流编排。"
            "要求精通 Python，有 LangChain、FAISS、Docker 等工具使用经验，"
            "了解 DeepSeek、GPT 等大模型 API 调用和 prompt engineering。"
            "硕士及以上学历，航空航天或计算机相关专业。"
        ),
        requirements=["Python", "LangChain", "FAISS", "Docker", "DeepSeek", "大模型", "RAG", "硕士"],
    ),
]
