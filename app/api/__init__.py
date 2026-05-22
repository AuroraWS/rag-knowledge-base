"""智能招聘助手 — API 路由层

包含五个核心路由模块：
- profile:     简历信息管理
- generate:    LLM 内容生成（自荐信、项目介绍、求职信）
- applications:投递记录 CRUD
- command:     自然语言命令解析
- recommend:   岗位推荐（JD 匹配）
"""

from app.api.profile import router as profile_router
from app.api.generate import router as generate_router
from app.api.applications import router as applications_router
from app.api.command import router as command_router
from app.api.recommend import router as recommend_router

__all__ = [
    "profile_router",
    "generate_router",
    "applications_router",
    "command_router",
    "recommend_router",
]
