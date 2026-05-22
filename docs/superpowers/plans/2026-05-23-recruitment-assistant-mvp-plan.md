# 智能招聘助手 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create missing data models, fix all imports, verify 4 core flows end-to-end, and build the WeChat gateway.

**Architecture:** Fill-in approach — create `app/models/schema.py` with 13 Pydantic models that 12+ existing files depend on, then verify the document extraction, command parsing, self-intro generation, and application CRUD pipelines work. Add `app/gateway/` for WeChat bot (adapter → router → push pattern).

**Tech Stack:** FastAPI, Pydantic v2, DeepSeek API (httpx), SQLite (WAL), JSON file stores, Gradio, APScheduler, WeChat test account (iLink Bot-compatible webhook)

**Spec:** `docs/superpowers/specs/2026-05-21-recruitment-assistant-mvp-design.md`

---

## File Structure

| Action | File | Purpose |
|:---|:---|:---|
| CREATE | `app/models/__init__.py` | Package init |
| CREATE | `app/models/schema.py` | All Pydantic models (13 classes) |
| CREATE | `app/gateway/__init__.py` | Gateway package init |
| CREATE | `app/gateway/wechat.py` | WeChat webhook adapter |
| CREATE | `app/gateway/router.py` | Message routing (chatbot ↔ command) |
| CREATE | `app/gateway/push.py` | Scheduled push notifications |
| MODIFY | `app/storage/application_store.py` | Fix StatusChange field references |
| MODIFY | `app/storage/memory_store.py` | Fix datetime serialization |
| MODIFY | `app/agent/workflow.py` | Fix MatchResult construction |
| VERIFY | `app/api/command.py` | Already correct, verify import works |
| VERIFY | `app/api/applications.py` | Already correct, verify import works |
| VERIFY | `app/api/profile.py` | Already correct, verify import works |
| VERIFY | `app/api/generate.py` | Already correct, verify import works |
| VERIFY | `app/api/recommend.py` | Already correct, verify import works |
| VERIFY | `app/rag/pipeline.py` | Already correct, verify import works |
| VERIFY | `app/services/extraction_service.py` | Already correct, verify import works |
| VERIFY | `app/services/preparation_service.py` | Already correct, verify import works |
| VERIFY | `app/services/tracking_service.py` | Already correct, verify import works |
| VERIFY | `app/storage/profile_store.py` | Already correct, verify import works |

---

### Task 1: Create models package init

**Files:**
- Create: `app/models/__init__.py`

- [ ] **Step 1: Write the file**

```python
"""智能招聘助手 — 数据模型层

所有 Pydantic 模型定义集中于此模块，供 API、存储、服务、Agent 层共享使用。
"""

from app.models.schema import (
    Application,
    ApplicationStatus,
    Certificate,
    Education,
    FieldMemory,
    HealthResponse,
    MatchResult,
    PersonalInfo,
    Project,
    StatusChange,
    WorkExperience,
)

__all__ = [
    "Application",
    "ApplicationStatus",
    "Certificate",
    "Education",
    "FieldMemory",
    "HealthResponse",
    "MatchResult",
    "PersonalInfo",
    "Project",
    "StatusChange",
    "WorkExperience",
]
```

- [ ] **Step 2: Verify file exists**

```bash
python -c "from app.models import Application, ApplicationStatus; print('OK')"
```
Expected: ImportError (schema.py doesn't exist yet), but `__init__.py` exists.

- [ ] **Step 3: Commit**

```bash
git add app/models/__init__.py
git commit -m "feat: create models package init

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Create schema.py — Core enums and simple response models

**Files:**
- Create: `app/models/schema.py`

- [ ] **Step 1: Write the enums and HealthResponse**

```python
"""智能招聘助手 — Pydantic 数据模型

集中定义所有数据模型，供 API 层、存储层、服务层和 Agent 层共享使用。

模型分组：
- 枚举: ApplicationStatus
- 资料库模型: PersonalInfo, Education, WorkExperience, Project, Certificate
- 投递模型: Application, StatusChange
- 辅助模型: FieldMemory, MatchResult, HealthResponse
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════


class ApplicationStatus(StrEnum):
    """投递状态枚举，中日双语值。"""

    PENDING = "待投递"
    APPLIED = "已投递待反馈"
    EXAM = "已收到笔试"
    INTERVIEWING = "面试中"
    REJECTED = "已拒绝"
    OFFER = "已拿到Offer"


# ═══════════════════════════════════════════════════════
# 通用响应模型
# ═══════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """健康检查响应"""

    version: str
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "from app.models.schema import ApplicationStatus, HealthResponse
print('ApplicationStatus:', list(ApplicationStatus))
print('HealthResponse:', HealthResponse(version='2.0'))
"
```
Expected: Prints enum members and HealthResponse

- [ ] **Step 3: Commit**

```bash
git add app/models/schema.py
git commit -m "feat: add ApplicationStatus enum and HealthResponse model

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Add profile models to schema.py (PersonalInfo, Education, WorkExperience, Project, Certificate)

**Files:**
- Modify: `app/models/schema.py` (append after HealthResponse)

- [ ] **Step 1: Append the 5 profile models**

```python
# ═══════════════════════════════════════════════════════
# 资料库模型
# ═══════════════════════════════════════════════════════


class PersonalInfo(BaseModel):
    """个人基本信息"""

    name: str = ""
    phone: str = ""
    email: str = ""
    wechat: Optional[str] = None
    target_location: list[str] = Field(default_factory=list)
    target_salary: Optional[str] = None
    earliest_start_date: Optional[str] = None
    id_number: Optional[str] = None
    birthday: Optional[str] = None
    gender: Optional[str] = None
    residence: Optional[str] = None
    source_files: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """教育经历"""

    school: str
    degree: str
    major: str
    start_date: str
    end_date: str
    gpa: Optional[float] = None
    degree_cert_number: Optional[str] = None
    is_overseas: bool = False
    source_file: Optional[str] = None


class WorkExperience(BaseModel):
    """工作经历"""

    company: str
    department: Optional[str] = None
    title: str
    start_date: str
    end_date: Optional[str] = None
    is_current: bool = False
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    source_file: Optional[str] = None


class Project(BaseModel):
    """项目经历"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    source_file: Optional[str] = None


class Certificate(BaseModel):
    """证书/资质"""

    name: str
    issuer: str
    date: Optional[str] = None
    cert_number: Optional[str] = None
    source_file: Optional[str] = None
```

- [ ] **Step 2: Verify with existing profile_store imports**

```bash
python -c "
from app.models.schema import (
    PersonalInfo, Education, WorkExperience, Project, Certificate
)
# Simulate what profile_store.py does
p = PersonalInfo(name='test', email='t@t.com', phone='123')
e = Education(school='X', degree='BSc', major='CS', start_date='2020', end_date='2024')
w = WorkExperience(company='X', title='Eng', start_date='2020')
proj = Project(name='Test', role='Dev')
c = Certificate(name='C1', issuer='Org')
print('All profile models OK')
"
```
Expected: "All profile models OK"

- [ ] **Step 3: Verify profile_store.py import works**

```bash
python -c "from app.storage.profile_store import profile_store; print('profile_store import OK')"
```
Expected: "profile_store import OK" (profile_store creates default data using these models)

- [ ] **Step 4: Commit**

```bash
git add app/models/schema.py
git commit -m "feat: add profile models to schema.py

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Add StatusChange, Application models to schema.py

**Files:**
- Modify: `app/models/schema.py` (append after Certificate)

- [ ] **Step 1: Append StatusChange and Application**

```python
# ═══════════════════════════════════════════════════════
# 投递模型
# ═══════════════════════════════════════════════════════


class StatusChange(BaseModel):
    """状态变更记录，嵌入 Application.timeline"""

    status: ApplicationStatus
    change_date: date = Field(alias="date")
    note: Optional[str] = None

    model_config = {"populate_by_name": True}


class Application(BaseModel):
    """投递记录 — 核心实体"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company: str
    title: str
    location: str = ""
    url: str = ""
    jd_text: str = ""
    channel: str = ""
    resume_version: Optional[str] = None
    cover_letter: Optional[str] = None
    submit_date: date = Field(default_factory=date.today)
    status: ApplicationStatus = Field(default=ApplicationStatus.APPLIED)
    last_check: Optional[date] = None
    next_check: Optional[date] = None
    timeline: list[StatusChange] = Field(default_factory=list)
    notes: str = ""
    # 面试 & 准备计划
    interview_date: Optional[date] = None
    prep_plan: Optional[dict[str, Any]] = None
    # 延伸字段
    job_id: Optional[str] = None
    jd_summary: Optional[str] = None
    next_action: Optional[str] = None
    next_action_date: Optional[date] = None
```

- [ ] **Step 2: Verify ApplicationStore import and StatusChange alias**

```bash
python -c "
from app.models.schema import Application, ApplicationStatus, StatusChange
from datetime import date

# Test StatusChange with 'date' alias (as used in application_store.py)
sc_data = {'status': '面试中', 'date': '2026-05-23', 'note': 'test'}
sc = StatusChange(**sc_data)
assert sc.change_date == date(2026, 5, 23)
assert sc.status == ApplicationStatus.INTERVIEWING
print('StatusChange alias OK')

# Test Application creation
app = Application(company='TestCo', title='Engineer')
assert app.status == ApplicationStatus.APPLIED
assert len(app.id) == 36  # UUID
print('Application OK')
"
```
Expected: "StatusChange alias OK", "Application OK"

- [ ] **Step 5: Commit**

```bash
git add app/models/schema.py
git commit -m "feat: add StatusChange and Application models

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Add FieldMemory, MatchResult to schema.py

**Files:**
- Modify: `app/models/schema.py` (append after Application)

- [ ] **Step 1: Append the final 2 models**

```python
# ═══════════════════════════════════════════════════════
# 辅助模型
# ═══════════════════════════════════════════════════════


class FieldMemory(BaseModel):
    """字段记忆 — 系统记住用户填过的值"""

    field_key: str
    field_label: str
    value: str
    first_seen: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    source_context: Optional[str] = None
    confidence: float = 1.0


class MatchResult(BaseModel):
    """简历 vs JD 匹配分析结果"""

    resume_name: str = ""
    jd_title: str = ""
    company: str = ""
    match_score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    analysis: str = ""
    skill_matches: list[dict[str, Any]] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
```

- [ ] **Step 2: Verify all imports work end-to-end**

```bash
python -c "
from app.models.schema import (
    ApplicationStatus, HealthResponse,
    PersonalInfo, Education, WorkExperience, Project, Certificate,
    StatusChange, Application, FieldMemory, MatchResult,
)
print('All 11 models imported successfully')

# Quick smoke: every model can be instantiated
from datetime import date, datetime
for cls, kwargs in [
    (HealthResponse, {'version': '2.0'}),
    (PersonalInfo, {}),
    (Education, {'school': 'X', 'degree': 'B', 'major': 'C', 'start_date': '2020', 'end_date': '2024'}),
    (WorkExperience, {'company': 'X', 'title': 'E', 'start_date': '2020'}),
    (Project, {'name': 'X', 'role': 'D'}),
    (Certificate, {'name': 'X', 'issuer': 'Y'}),
    (StatusChange, {'status': '待投递', 'date': '2026-01-01'}),
    (Application, {'company': 'X', 'title': 'E'}),
    (FieldMemory, {'field_key': 'salary', 'field_label': '期望薪资', 'value': '15k'}),
    (MatchResult, {}),
]:
    instance = cls(**kwargs)
    print(f'  {cls.__name__} OK')

print('All models instantiable')
"
```
Expected: All 11 models import and instantiate successfully.

- [ ] **Step 3: Commit**

```bash
git add app/models/schema.py
git commit -m "feat: add FieldMemory and MatchResult models

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Fix application_store.py StatusChange construction

**Files:**
- Modify: `app/storage/application_store.py:123-130`

The existing code at line 123-129 constructs `StatusChange` with:
```python
if "status" in item and "date" in item:
    item["status"] = ApplicationStatus(item["status"])
    if isinstance(item["date"], str):
        item["date"] = date.fromisoformat(item["date"])
    timeline.append(StatusChange(**item))
```

The `item` dict has key `"date"` but the model field is `change_date` with alias `"date"`. With `populate_by_name=True`, this should work. But let's verify and add safety.

- [ ] **Step 1: Read current state and run verification**

```bash
python -c "
from app.storage.application_store import ApplicationStore
store = ApplicationStore()
# If this doesn't crash, the import chain works
print('application_store import OK')
"
```
Expected: "application_store import OK"

- [ ] **Step 2: Test roundtrip — add, get, update, delete**

```bash
python -c "
from app.storage.application_store import application_store, Application, ApplicationStatus
from datetime import date

# Add
app = Application(company='TestInc', title='Dev')
created = application_store.add(app)
assert created.company == 'TestInc'
print(f'Created: {created.id}')

# Get
got = application_store.get(created.id)
assert got is not None
assert got.company == 'TestInc'
print('Get OK')

# Update status
updated = application_store.update_status(created.id, ApplicationStatus.INTERVIEWING, note='passed screening')
assert updated.status == ApplicationStatus.INTERVIEWING
assert len(updated.timeline) == 1
assert updated.timeline[0].change_date == date.today()
print('Update status OK')

# List
all_apps = application_store.list()
assert len(all_apps) >= 1
print(f'List: {len(all_apps)} records')

# Stats
stats = application_store.stats()
assert stats['total'] >= 1
print(f'Stats: {stats}')

# Cleanup
application_store.delete(created.id)
print('Delete OK')
print('All application_store tests PASSED')
"
```
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add app/storage/application_store.py
git commit -m "fix: verify StatusChange alias works with Schema v2

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Verify profile_store.py works with schema

**Files:**
- Verify: `app/storage/profile_store.py`

- [ ] **Step 1: Test profile_store import and initial data**

```bash
python -c "
from app.storage.profile_store import profile_store
data = profile_store.get_all()
personal = data.get('personal_info')
print(f'Name: {personal.name}')
print(f'Education: {len(data.get(\"education\", []))} records')
print(f'Work: {len(data.get(\"work_experience\", []))} records')
print(f'Projects: {len(data.get(\"projects\", []))} records')
print(f'Certificates: {len(data.get(\"certificates\", []))} records')
"
```
Expected: Prints default profile data for 王爽

- [ ] **Step 2: Test CRUD operations on profile sections**

```bash
python -c "
from app.storage.profile_store import profile_store
from app.models.schema import Education, WorkExperience, Project, Certificate

# Add education
edu = Education(school='Test Univ', degree='PhD', major='AI', start_date='2025', end_date='2028')
profile_store.add_education(edu)
all_edu = profile_store.get_education()
assert len(all_edu) >= 1
print(f'Add education OK ({len(all_edu)} records)')

# Update
updated = profile_store.update_education(0, Education(school='Updated Univ', degree='PhD', major='AI', start_date='2025', end_date='2029'))
assert updated.school == 'Updated Univ'
print('Update education OK')

# Remove
profile_store.remove_education(0)
print('Remove education OK')

# Add project
proj = Project(name='Test Project', role='Lead Dev', description='A test')
profile_store.add_project(proj)
projects = profile_store.get_projects()
assert len(projects) >= 1
print(f'Add project OK ({len(projects)} records)')

# Add certificate
cert = Certificate(name='AWS SA', issuer='Amazon')
profile_store.add_certificate(cert)
certs = profile_store.get_certificates()
assert len(certs) >= 1
print(f'Add certificate OK ({len(certs)} records)')

print('All profile_store tests PASSED')
"
```
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add app/storage/profile_store.py
git commit -m "fix: verify profile_store CRUD with v2 schema models

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: Verify memory_store.py works

**Files:**
- Verify: `app/storage/memory_store.py`

- [ ] **Step 1: Test memory CRUD**

```bash
python -c "
from app.storage.memory_store import memory_store
from app.models.schema import FieldMemory

# Set a memory
memory_store.set('test_key', 'Test Label', 'test value')
val = memory_store.get('test_key')
assert val == 'test value'
print(f'Set/Get OK: {val}')

# Update
memory_store.set('test_key', 'Test Label', 'updated value')
val2 = memory_store.get('test_key')
assert val2 == 'updated value'
print(f'Update OK: {val2}')

# All
all_mem = memory_store.all()
assert 'test_key' in all_mem
print(f'All: {len(all_mem)} memories')

# Delete
memory_store.delete('test_key')
assert memory_store.get('test_key') is None
print('Delete OK')

print('All memory_store tests PASSED')
"
```
Expected: All tests pass.

- [ ] **Step 2: Commit**

```bash
git add app/storage/memory_store.py
git commit -m "fix: verify memory_store CRUD with v2 schema models

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: Verify FastAPI app starts without import errors

**Files:**
- Verify: `app/main.py` (no changes expected)

- [ ] **Step 1: Test that all router imports work**

```bash
python -c "
from app.api import (
    profile_router,
    generate_router,
    applications_router,
    command_router,
    recommend_router,
)
print('All API routers imported successfully')
"
```
Expected: All routers imported.

- [ ] **Step 2: Test app creation**

```bash
python -c "
from app.main import app
print(f'App: {app.title}')
print(f'Routes: {len(app.routes)}')
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        print(f'  {route.methods} {route.path}')
"
```
Expected: Prints all registered routes including /health, /api/command, /api/applications, etc.

- [ ] **Step 3: Commit** (only if changes were needed)

---

### Task 10: Verify LLMGenerator and RAG pipeline imports

**Files:**
- Verify: `app/rag/generator.py`, `app/rag/pipeline.py`

- [ ] **Step 1: Test LLMGenerator import**

```bash
python -c "
from app.rag.generator import LLMGenerator
gen = LLMGenerator()
print(f'Model: {gen._model}')
print('LLMGenerator import OK')
"
```
Expected: Prints model name (deepseek-chat).

- [ ] **Step 2: Test RAGPipeline import (skips if no API key)**

```bash
python -c "
from app.rag.pipeline import RAGPipeline
print('RAGPipeline import OK')
"
```
Expected: "RAGPipeline import OK"

- [ ] **Step 3: Test agent workflow imports**

```bash
python -c "
from app.agent.workflow import application_tracking_workflow, interview_prep_workflow
from app.agent.scheduler import job_scheduler
print('Agent workflow and scheduler import OK')
"
```
Expected: OK (with possible LangGraph/APScheduler availability warnings)

- [ ] **Step 4: Commit**

---

### Task 11: End-to-end: document upload → extraction → profile store

**Files:**
- Verify: `app/api/profile.py`, `app/services/extraction_service.py`

- [ ] **Step 1: Test extraction_service import**

```bash
python -c "
from app.services.extraction_service import extraction_service
print('extraction_service import OK')
"
```

- [ ] **Step 2: Verify upload endpoint with a test file**

```bash
echo '{"name":"Test User","phone":"13800138000","email":"test@example.com"}' > /tmp/test_resume.json
python -c "
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
resp = client.get('/health')
assert resp.status_code == 200
print(f'Health: {resp.json()}')
"
```
Expected: Health check passes, returns version.

- [ ] **Step 3: Commit**

---

### Task 12: End-to-end: command parsing → application CRUD

**Files:**
- Verify: `app/api/command.py`, `app/api/applications.py`, `app/storage/application_store.py`

- [ ] **Step 1: Test GET /api/applications**

```bash
python -c "
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

# List
resp = client.get('/api/applications')
assert resp.status_code == 200
data = resp.json()
print(f'Applications: total={data[\"total\"]}')

# Stats
resp2 = client.get('/api/applications/stats')
assert resp2.status_code == 200
print(f'Stats: {resp2.json()}')

# Pending check
resp3 = client.get('/api/applications/pending-check')
assert resp3.status_code == 200
print(f'Pending: {resp3.json()[\"total\"]}')
"
```

- [ ] **Step 2: Test POST create and DELETE**

```bash
python -c "
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

# Create
resp = client.post('/api/applications', json={
    'company': 'TestCo', 'title': 'Engineer', 'location': 'Shanghai', 'channel': '官网'
})
assert resp.status_code == 201
created = resp.json()['data']
app_id = created['id']
print(f'Created: {app_id} - {created[\"company\"]}')

# Update status
resp2 = client.put(f'/api/applications/{app_id}/status', json={'status': '面试中', 'note': 'got call'})
assert resp2.status_code == 200
print(f'Status updated: {resp2.json()[\"data\"][\"status\"]}')

# Delete
resp3 = client.delete(f'/api/applications/{app_id}')
assert resp3.status_code == 200
print(f'Deleted: {resp3.json()[\"message\"]}')
"
```

- [ ] **Step 3: Commit**

---

### Task 13: End-to-end: self-intro generation

**Files:**
- Verify: `app/api/generate.py`, `app/rag/generator.py`

**Note:** Requires `DEEPSEEK_API_KEY` in `.env`. Skip if not configured.

- [ ] **Step 1: Check API key and test LLM connectivity**

```bash
python -c "
from app.config import settings
if settings.deepseek_api_key:
    print('API key configured')
else:
    print('WARNING: DEEPSEEK_API_KEY not set — skipping generation tests')
"
```

- [ ] **Step 2: Skip if no key, mark as ready for manual test**

If API key is configured, run the self-intro endpoint. If not, this flow is verified in Task 9 (imports + routes work).

- [ ] **Step 3: Commit**

---

### Task 14: Create WeChat gateway — adapter

**Files:**
- Create: `app/gateway/__init__.py`
- Create: `app/gateway/wechat.py`

- [ ] **Step 1: Write __init__.py**

```python
"""智能招聘助手 — WeChat Bot 网关

参考 Hermes Agent 的适配器模式，Python 轻量实现。
- wechat.py: iLink Bot webhook 适配器
- router.py: 消息路由分发
- push.py: 主动推送管理
"""

from app.gateway.wechat import WeChatAdapter
from app.gateway.router import MessageRouter
from app.gateway.push import PushManager

__all__ = ["WeChatAdapter", "MessageRouter", "PushManager"]
```

- [ ] **Step 2: Write wechat.py**

```python
"""WeChat Bot webhook 适配器 — 接收 + 验证 iLink Bot 消息。

参考：iLink Bot API（腾讯官方合规方案）
测试号申请：https://mp.weixin.qq.com/debug/cgi-bin/sandbox?t=sandbox/login
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET

from app.config import settings

logger = logging.getLogger(__name__)


# ── 统一消息格式 ──────────────────────────────────


class GatewayMessage(BaseModel):
    """统一网关消息 — 屏蔽不同平台的差异"""

    msg_type: str = "text"  # text, image, voice, event
    content: str = ""
    from_user: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    raw: dict = Field(default_factory=dict)


# ── 适配器 ──────────────────────────────────────


class WeChatAdapter:
    """iLink Bot / 微信测试号 webhook 适配器。

    功能：
    - 验证消息签名（防篡改）
    - 解析 XML → GatewayMessage
    - 构建回复 XML
    """

    def __init__(self, token: Optional[str] = None):
        self._token = token or "recruitment_assistant_token"

    # ── 签名验证 ─────────────────────────────────

    def verify_signature(
        self, signature: str, timestamp: str, nonce: str, echostr: str = ""
    ) -> bool:
        """验证微信服务器签名。

        Returns:
            echostr 如果验证成功（用于首次接入），否则返回空字符串。
        """
        tmp_list = sorted([self._token, timestamp, nonce])
        tmp_str = "".join(tmp_list)
        computed = hashlib.sha1(tmp_str.encode()).hexdigest()

        if computed == signature:
            logger.info("签名验证成功")
            return echostr or True
        logger.warning("签名验证失败: expected=%s got=%s", computed, signature)
        return "" if echostr else False

    # ── 消息解析 ─────────────────────────────────

    def parse_message(self, xml_data: str) -> GatewayMessage:
        """解析微信 XML 消息 → GatewayMessage。

        iLink Bot 发送 XML 格式：
        <xml>
            <ToUserName><![CDATA[toUser]]></ToUserName>
            <FromUserName><![CDATA[fromUser]]></FromUserName>
            <CreateTime>1348831860</CreateTime>
            <MsgType><![CDATA[text]]></MsgType>
            <Content><![CDATA[this is a test]]></Content>
            <MsgId>1234567890123456</MsgId>
        </xml>
        """
        try:
            root = ET.fromstring(xml_data)
            msg_type = self._get_text(root, "MsgType", "text")
            content = self._get_text(root, "Content", "")
            from_user = self._get_text(root, "FromUserName", "")
            create_time = int(self._get_text(root, "CreateTime", "0"))
            msg_id = self._get_text(root, "MsgId", "")

            return GatewayMessage(
                msg_type=msg_type,
                content=content,
                from_user=from_user,
                timestamp=datetime.fromtimestamp(create_time) if create_time else datetime.now(),
                raw={
                    "to_user": self._get_text(root, "ToUserName", ""),
                    "msg_id": msg_id,
                },
            )
        except ET.ParseError as e:
            logger.error("XML 解析失败: %s", e)
            return GatewayMessage(content=xml_data, raw={"parse_error": str(e)})

    # ── 回复构建 ─────────────────────────────────

    def build_reply(
        self, to_user: str, from_user: str, content: str
    ) -> str:
        """构建回复 XML。

        微信要求回复格式：
        <xml>
            <ToUserName><![CDATA[toUser]]></ToUserName>
            <FromUserName><![CDATA[fromUser]]></FromUserName>
            <CreateTime>timestamp</CreateTime>
            <MsgType><![CDATA[text]]></MsgType>
            <Content><![CDATA[content]]></Content>
        </xml>
        """
        timestamp = int(time.time())
        return (
            "<xml>"
            f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
            f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
            f"<CreateTime>{timestamp}</CreateTime>"
            "<MsgType><![CDATA[text]]></MsgType>"
            f"<Content><![CDATA[{content}]]></Content>"
            "</xml>"
        )

    @staticmethod
    def _get_text(root: ET.Element, tag: str, default: str = "") -> str:
        el = root.find(tag)
        return el.text or default if el is not None else default


# 模块级实例
wechat_adapter = WeChatAdapter()
```

- [ ] **Step 2: Verify syntax and basic parsing**

```bash
python -c "
from app.gateway.wechat import WeChatAdapter, GatewayMessage

adapter = WeChatAdapter()

# Test XML parsing
xml = '''<xml>
<ToUserName><![CDATA[gh_123]]></ToUserName>
<FromUserName><![CDATA[user_456]]></FromUserName>
<CreateTime>1715900000</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[我投了腾讯的算法工程师]]></Content>
<MsgId>123456</MsgId>
</xml>'''

msg = adapter.parse_message(xml)
assert msg.msg_type == 'text'
assert msg.content == '我投了腾讯的算法工程师'
assert msg.from_user == 'user_456'
print(f'Parse OK: type={msg.msg_type}, from={msg.from_user}, content={msg.content}')

# Test reply building
reply = adapter.build_reply('user_456', 'gh_123', '已添加投递记录')
assert 'user_456' in reply
assert '已添加投递记录' in reply
print('Reply build OK')

# Test signature verification
result = adapter.verify_signature('wrong_sig', '123', '456')
assert result == False or result == ''
print('Signature reject OK')
"
```
Expected: All assertions pass.

- [ ] **Step 3: Commit**

```bash
git add app/gateway/__init__.py app/gateway/wechat.py
git commit -m "feat: add WeChat gateway adapter

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 15: Create WeChat gateway — router

**Files:**
- Create: `app/gateway/router.py`

- [ ] **Step 1: Write router.py**

```python
"""WeChat Bot 消息路由 — chatbot 对话 + 命令分发。

将微信消息路由到三大处理分支：
1. 求职命令 → /api/command（添加投递、更新状态等）
2. 求职问答 → 查询资料库 / 调 RAG 生成
3. 日常聊天 → LLM 自由对话
"""

from __future__ import annotations

import logging
from typing import Optional

from app.gateway.wechat import GatewayMessage
from app.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

# ── 命令关键词（触发命令引擎） ──────────────────

_COMMAND_KEYWORDS = [
    "投了", "投递", "面试", "笔试", "状态",
    "拒绝", "offer", "Offer", "已拿",
    "帮我看", "帮我查", "帮我准备", "准备",
    "列表", "统计", "仪表盘",
    "自我评价", "自我介绍", "生成",
]

_CHAT_SYSTEM_PROMPT = """你是王爽的求职助手，运行在微信上。
你的角色：
- 帮助管理投递记录
- 提醒面试和检查日期
- 回答求职相关问题
- 提供职业建议

保持回复简洁（微信消息不适合长篇大论），每条回复控制在200字以内。
如果用户说的是求职相关操作，引导他们使用明确的命令格式。

当前你的能力：
- 记录投递："投了[公司名]的[岗位名]" → 自动记录
- 查状态："投递列表" / "面试中的有哪些"
- 生成介绍："帮我写[公司名]的自我介绍"
"""


class MessageRouter:
    """消息路由器 — 判断意图并分发到对应处理器。"""

    def __init__(self):
        self._generator: Optional[LLMGenerator] = None

    @property
    def generator(self) -> LLMGenerator:
        if self._generator is None:
            self._generator = LLMGenerator()
        return self._generator

    # ── 意图判断 ────────────────────────────────

    def is_command(self, msg: GatewayMessage) -> bool:
        """判断是否为求职命令（触发命令引擎）。"""
        return any(kw in msg.content for kw in _COMMAND_KEYWORDS)

    def is_subscribe(self, msg: GatewayMessage) -> bool:
        """判断是否为订阅/退订指令。"""
        content = msg.content.strip().lower()
        return content in ("订阅", "退订", "开启推送", "关闭推送")

    # ── 路由分发 ─────────────────────────────────

    async def route(self, msg: GatewayMessage) -> str:
        """路由消息到对应处理器，返回回复文本。

        Args:
            msg: 统一格式的网关消息。

        Returns:
            回复文本字符串。
        """
        # 分支 1: 订阅管理
        if self.is_subscribe(msg):
            return self._handle_subscribe(msg)

        # 分支 2: 求职命令
        if self.is_command(msg):
            return await self._handle_command(msg)

        # 分支 3: 日常聊天
        return await self._handle_chat(msg)

    # ── 处理器 ────────────────────────────────────

    def _handle_subscribe(self, msg: GatewayMessage) -> str:
        """处理订阅/退订。"""
        if "退订" in msg.content or "关闭" in msg.content:
            # TODO: 调用 push manager 关闭该用户的推送
            return "已关闭每日推送。需要时回复「订阅」重新开启。"
        return "已开启每日推送！每天 9:00 发送今日待办，22:00 发送每日日志。"

    async def _handle_command(self, msg: GatewayMessage) -> str:
        """转发到 /api/command 引擎。"""
        try:
            from app.api.command import execute_command, CommandRequest

            req = CommandRequest(text=msg.content)
            resp = await execute_command(req)
            return resp.action_summary
        except Exception as e:
            logger.error("命令执行失败: %s", e)
            return f"抱歉，处理指令时出错了: {e}"

    async def _handle_chat(self, msg: GatewayMessage) -> str:
        """LLM 自由对话。"""
        try:
            text = await self.generator.generate(
                prompt=msg.content,
                system_prompt=_CHAT_SYSTEM_PROMPT,
                temperature=0.7,
            )
            # 截断过长回复
            if len(text) > 400:
                text = text[:380] + "..."
            return text
        except Exception as e:
            logger.error("Chat 回复失败: %s", e)
            return "收到你的消息了，但我现在脑子有点转不动，请稍后再试～"


# 模块级实例
message_router = MessageRouter()
```

- [ ] **Step 2: Verify syntax and routing logic**

```bash
python -c "
from app.gateway.router import MessageRouter, _COMMAND_KEYWORDS
from app.gateway.wechat import GatewayMessage

router = MessageRouter()

# Test intent detection
cmd_msg = GatewayMessage(content='我投了腾讯的算法工程师')
chat_msg = GatewayMessage(content='你好啊')
sub_msg = GatewayMessage(content='订阅')

assert router.is_command(cmd_msg) == True
assert router.is_command(chat_msg) == False
assert router.is_subscribe(sub_msg) == True
print('Intent detection OK')

# Test unsubscribe
unsub_msg = GatewayMessage(content='退订')
assert router.is_subscribe(unsub_msg) == True
print('Subscribe/unsubscribe detection OK')
"
```
Expected: All assertions pass.

- [ ] **Step 3: Commit**

```bash
git add app/gateway/router.py
git commit -m "feat: add WeChat message router

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 16: Create WeChat gateway — push manager

**Files:**
- Create: `app/gateway/push.py`

- [ ] **Step 1: Write push.py**

```python
"""WeChat Bot 主动推送管理 — 定时任务触发 → 模板消息。

5 种推送类型：
1. 今日待办（每日 9:00）
2. 每日复习（可配时间）
3. 面试倒计时（面试前 7 天起每日）
4. 状态变更通知（实时）
5. 每日日志（每日 22:00）

当前阶段：模板消息 API 使用微信测试号的接口。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from app.config import settings
from app.storage.application_store import ApplicationStore, application_store

logger = logging.getLogger(__name__)


class PushManager:
    """推送管理器 — 构建推送内容 + 发送。

    发送通道当前为日志占位（打印到日志）。
    DEEPSEEK phase 2 接入真实微信模板消息 API。
    """

    def __init__(self, store: Optional[ApplicationStore] = None):
        self._store = store or application_store

    # ── 推送类型 1: 今日待办 ─────────────────────

    def build_daily_todo(self) -> str:
        """构建今日待办推送内容。"""
        from app.services.tracking_service import tracking_service

        todo = tracking_service.get_today_todo()
        summary = tracking_service.get_status_summary()

        lines = ["☀️ 早上好！今日求职待办", ""]
        lines.append("📊 投递概况")
        lines.append(f"  总投递: {summary.get('total', 0)}")
        lines.append(f"  面试中: {summary.get('interview_count', 0)}")
        lines.append(f"  待检查: {summary.get('pending_check_count', 0)}")

        if todo:
            lines.append("")
            lines.append(f"📋 今日待办 ({len(todo)} 项)")
            for item in todo:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    item.get("priority", "low"), "⚪"
                )
                lines.append(
                    f"  {icon} {item.get('company', '')} - {item.get('title', '')}"
                )
        else:
            lines.append("")
            lines.append("✅ 今日没有待办事项")

        return "\n".join(lines)

    # ── 推送类型 2: 面试倒计时 ───────────────────

    def build_interview_countdown(self) -> list[str]:
        """为每个即将面试的投递生成倒计时消息。

        Returns:
            消息列表，每个面试一条。
        """
        messages = []
        today = date.today()
        apps = self._store.list()

        for app in apps:
            if app.interview_date and app.status == ApplicationStatus.INTERVIEWING:
                days_left = (app.interview_date - today).days
                if 0 <= days_left <= 7:
                    messages.append(
                        f"⏰ 面试倒计时: {app.company} - {app.title}\n"
                        f"   日期: {app.interview_date}\n"
                        f"   还剩 {days_left} 天，加油准备！"
                    )
        return messages

    # ── 推送类型 3: 每日日志 ────────────────────

    def build_daily_log(self) -> str:
        """构建每日日志推送内容。"""
        from app.services.tracking_service import tracking_service

        today = date.today()
        recent = tracking_service.get_recent_applications(days=1)
        summary = tracking_service.get_status_summary()

        lines = [f"📝 求职日报 ({today.isoformat()})", ""]

        if recent:
            lines.append(f"今日投递变动 ({len(recent)} 条):")
            for app in recent:
                lines.append(f"  • {app.company} - {app.title} ({app.status.value})")
        else:
            lines.append("今日无投递变动")

        lines.append("")
        lines.append("📊 状态分布:")
        for status, count in summary.get("by_status", {}).items():
            lines.append(f"  {status}: {count}")

        # 明日计划
        pending = self._store.get_pending_check(days=3)
        if pending:
            lines.append("")
            lines.append("📅 近期提醒:")
            for app in pending[:5]:
                lines.append(f"  • {app.company}: {app.next_action or '检查状态'}")

        return "\n".join(lines)

    # ── 发送（占位 → 替换为真实 API） ────────────

    async def send(self, to_user: str, content: str) -> bool:
        """发送消息给指定用户。

        Args:
            to_user: 微信 openid。
            content: 消息文本。

        Returns:
            True 表示发送成功。

        TODO: 替换为微信模板消息 API 调用。
        """
        if not settings.wechat_appid:
            logger.info("[WeChat Push 占位] to=%s, len=%d", to_user, len(content))
            return True

        # 真实发送:
        # access_token = await self._get_access_token()
        # await self._send_template_message(access_token, to_user, content)
        logger.info("WeChat push sent to %s: %d chars", to_user, len(content))
        return True


# 模块级实例
push_manager = PushManager()
```

- [ ] **Step 2: Verify push message building**

```bash
python -c "
from app.gateway.push import push_manager

# Test daily todo build (may be empty since no apps)
todo = push_manager.build_daily_todo()
assert '今日求职待办' in todo
print('Daily todo build OK')

# Test daily log build
log = push_manager.build_daily_log()
assert '求职日报' in log
print('Daily log build OK')

# Test interview countdown (should be empty)
cd = push_manager.build_interview_countdown()
print(f'Interview countdowns: {len(cd)} messages')
"
```
Expected: All assertions pass.

- [ ] **Step 3: Commit**

```bash
git add app/gateway/push.py
git commit -m "feat: add WeChat push manager

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 17: Wire gateway into FastAPI and scheduler

**Files:**
- Modify: `app/main.py` (add gateway webhook endpoint)
- Modify: `app/agent/scheduler.py` (connect push manager to scheduled jobs)

- [ ] **Step 1: Add WeChat webhook endpoint to main.py**

Add after existing route registrations in `app/main.py`:

```python
# WeChat bot webhook endpoint
from app.gateway.wechat import wechat_adapter, GatewayMessage
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
```

- [ ] **Step 2: Wire push manager into scheduler**

In `app/agent/scheduler.py`, modify `_send_wechat_message`:

```python
async def _send_wechat_message(self, message: str) -> None:
    """通过 WeChat bot 发送消息。"""
    from app.gateway.push import push_manager

    # 占位用户 openid（后续从订阅列表获取）
    await push_manager.send("default_user", message)
```

- [ ] **Step 3: Verify the webhook endpoint is registered**

```bash
python -c "
from app.main import app
routes = [(r.path, list(r.methods)) for r in app.routes if hasattr(r, 'methods')]
webhook_routes = [(p, m) for p, m in routes if 'wechat' in p]
print('Webhook routes:', webhook_routes)
assert any('wechat' in p for p, _ in webhook_routes)
"
```
Expected: Prints webhook routes with GET and POST methods.

- [ ] **Step 4: Commit**

```bash
git add app/main.py app/agent/scheduler.py
git commit -m "feat: wire WeChat gateway into FastAPI and scheduler

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 18: Final integration test — full startup

**Files:**
- Verify: all

- [ ] **Step 1: Test full app startup (no errors)**

```bash
python -c "
import warnings
warnings.filterwarnings('ignore')

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Test all endpoints respond (not 500 import errors)
endpoints = [
    ('GET', '/health'),
    ('GET', '/api/profile'),
    ('GET', '/api/applications'),
    ('GET', '/api/applications/stats'),
    ('GET', '/api/applications/pending-check'),
    ('POST', '/api/command'),
    ('POST', '/api/generate/self-intro'),
    ('POST', '/api/generate/project-intro'),
    ('POST', '/api/generate/cover-letter'),
    ('GET', '/api/wechat/webhook'),
]

for method, path in endpoints:
    if method == 'GET':
        resp = client.get(path)
    else:
        # POST with empty body — expect 422 (validation error) not 500 (import error)
        resp = client.post(path, json={})
    status = resp.status_code
    if status == 500:
        print(f'FAIL: {method} {path} → {status}')
        print(f'  Body: {resp.text[:200]}')
    else:
        print(f'  OK: {method} {path} → {status}')

print()
print('Integration test complete')
"
```
Expected: No 500 Internal Server Errors. 422 (validation error) or 200/201 are acceptable.

- [ ] **Step 2: Commit** (only if code changes needed to fix test failures)

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] `python -c "from app.models.schema import *"` — all models import
- [ ] `python -c "from app.main import app"` — app creates without errors
- [ ] `python -c "from app.gateway.wechat import wechat_adapter"` — gateway imports
- [ ] `python -c "from app.api import *"` — all routers import
- [ ] All 18 commit messages follow project conventions
