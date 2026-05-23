# 智能招聘助手 v2.0 — MVP 设计方案

> 日期：2026-05-21
> 版本：v2.0-mvp-draft
> 基于：[产品设计文档](../../../智能招聘助手v2-产品设计.md) | [需求规格说明书](../../../需求规格说明书.md)

---

## 1. 方案选择

**策略：填空优先（方案二）** — 保留现有代码架构，创建缺失的数据模型，修复导入并验证核心链路，然后搭建微信网关。

现有代码仓库已有 34 个文件 6000+ 行代码，包含完整的 API 路由、RAG 管线、Agent 调度、存储层和 Gradio 前端。唯一阻塞项是 `app/models/schema.py` 缺失（13 个文件依赖它）。

## 2. 架构设计

### 2.1 新建模块

| 模块 | 文件 | 用途 |
|:----|:----|:----|
| 数据模型 | `app/models/schema.py` | 13 个 Pydantic 模型，所有模块的共享数据源 |
| 微信网关 | `app/gateway/wechat.py` | iLink Bot webhook 适配器，消息接收与签名验证 |
| 微信网关 | `app/gateway/router.py` | 消息路由（chatbot 对话 ↔ 命令分发） |
| 微信网关 | `app/gateway/push.py` | 主动推送管理（模板消息） |

### 2.2 数据流

```
文档上传 → DeepSeek API 解析 → 规则+LLM 提取 → profile_store.json
自然语言 → /api/command → LLM 意图解析 → 分发执行 → SQLite / 生成器
投递管理 → SQLite CRUD → 仪表盘统计 → 定时检查提醒
微信消息 → iLink webhook → gateway → 命令引擎 / LLM 对话 → 回复/推送
```

## 3. 数据模型

### 3.1 资料库模型（5 个）

```python
class PersonalInfo(BaseModel):
    name, phone, email, wechat, target_location, target_salary,
    earliest_start_date, id_number, birthday, gender, residence, source_files

class Education(BaseModel):
    school, degree, major, start_date, end_date, gpa, degree_cert_number,
    is_overseas, source_file

class WorkExperience(BaseModel):
    company, department, title, start_date, end_date,
    responsibilities[], achievements[], tech_stack[], source_file

class Project(BaseModel):
    name, role, start_date, end_date, description,
    tech_stack[], highlights[], source_file

class Certificate(BaseModel):
    name, issuer, date, cert_number, source_file
```

### 3.2 投递 & 辅助模型（8 个）

```python
class ApplicationStatus(enum.StrEnum):
    待投递, 已投递待反馈, 已收到笔试, 面试中, 已拒绝, 已拿到Offer

class StatusChange(BaseModel):
    status: ApplicationStatus
    change_date: date  # alias "date" for backwards compat
    note: Optional[str]

class Application(BaseModel):
    id, company, title, location, url, jd_text, channel, resume_version,
    cover_letter, submit_date, status, last_check, next_check, timeline[],
    notes, interview_date, prep_plan, job_id, jd_summary, next_action, next_action_date

class FieldMemory(BaseModel):
    field_key, field_label, value, first_seen, last_updated, source_context, confidence

class MatchResult(BaseModel):
    score, skill_matches[], skill_gaps[], summary, recommendations[]

class HealthResponse, ProfileResponse  # 通用响应包装
```

### 3.3 引用关系

13 个文件依赖 `app.models.schema`：main.py, api/command.py, api/applications.py, api/profile.py, api/generate.py, storage/application_store.py, storage/profile_store.py, storage/memory_store.py, services/extraction_service.py, services/tracking_service.py, services/preparation_service.py, rag/pipeline.py, agent/workflow.py

## 4. 微信网关设计

参考 Hermes Agent 的适配器模式，Python 轻量实现。

### 4.1 消息接收

```
微信用户 → iLink Bot API → POST webhook → app/gateway/wechat.py
                                              │
                                    验证签名 + 解析
                                              │
                                              ▼
                                     app/gateway/router.py
                                      │           │
                              ┌───────┘           └────────┐
                              ▼                            ▼
                      Chat 消息                        Command 消息
                   → LLM 对话回复                  → /api/command 流程
                     (DeepSeek)                       (投递/查询/生成)
```

### 4.2 主动推送（5 种）

| 推送类型 | 内容 | 触发 |
|:--------|:----|:----|
| 今日待办 | 待检查投递 + 即将面试 | 每日 9:00 |
| 每日复习 | 准备计划中的今日知识点 | 每日可配 |
| 面试倒计时 | X天后 + 今日准备内容 | 面试前7天起每日 |
| 状态变更 | 新笔试/面试通知 | 用户操作触发 |
| 每日日志 | PDF 日报推送 | 每日 22:00 |

### 4.3 关键技术决策

- MVP 使用微信测试号（无需认证，有 webhook + 模板消息）
- 统一消息格式 `GatewayMessage {msg_type, content, from_user, timestamp}`
- 部署在 OCI ARM 实例上，Nginx 反代 + Let's Encrypt HTTPS
- 预计 4 个文件，~400 行 Python

## 5. MVP 范围

### 包含

| 功能 | 交付标准 |
|:----|:--------|
| 文档解析 | PDF 简历上传 → 字段提取（姓名/电话/邮箱/教育/技能/项目） |
| 资料库 CRUD | 查看 + 编辑 + 删除所有提取字段 |
| 自我评价生成 | 输入 JD → 生成 3 版本（精简/完整/英文） |
| 项目介绍生成 | 选择项目 → STAR 法则生成 |
| 投递管理 | 自然语言指令 + 完整 CRUD + 仪表盘 |
| 字段记忆 | 记住用户填过的自定义字段 |
| 微信 Chatbot | 聊天对话 + 命令执行 |
| 微信推送 | 今日待办 + 每日日志 |
| 前端 | Gradio 5-tab UI |

### 不含

- 浏览器插件、IMAP 邮箱检查、智能岗位推荐（P1）、OCR 图片识别（P4）、多用户

## 6. 实施步骤

### Day 1 — 补齐骨架
1.1 创建 `app/models/schema.py`（13 个 Pydantic 模型）
1.2 创建 `app/models/__init__.py`
1.3 修复所有模块中与模型不匹配的调用
1.4 `uvicorn app.main:app` 能启动
1.5 `pytest` 摸底现有测试

### Day 2 — 核心链路验证
2.1 文档上传 → 字段提取 → 入库
2.2 自然语言指令 → 意图解析 → 投递 CRUD
2.3 输入 JD → 生成自我评价
2.4 Gradio 前端 5 个 tab 可操作
2.5 定时任务（APScheduler）

### Day 3 — 微信网关
3.1 `app/gateway/wechat.py` — webhook 适配器
3.2 `app/gateway/router.py` — 消息路由
3.3 `app/gateway/push.py` — 主动推送
3.4 微信测试号配置 + webhook 联调

### Day 4 — 集成 & 联调
4.1 微信 Chat 对话 + 命令执行
4.2 定时推送验证
4.3 完整端到端测试
4.4 Gradio 微信配置页

## 7. 使用 Skills

| 阶段 | Skill | 用途 |
|:----|:----|:----|
| 设计 | brainstorming → writing-plans | 当前流程 |
| 实施 | test-driven-development | 写代码前先写测试 |
| 实施 | executing-plans | 按计划分步执行 |
| 调试 | investigate | 根因分析 |
| 验证 | verification-before-completion | 每步验证 |
| 微信 | Context7 | iLink Bot / WeChat API 文档 |
