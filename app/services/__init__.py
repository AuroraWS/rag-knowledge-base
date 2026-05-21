"""智能招聘助手 — 业务逻辑层（Services）

位于 API 路由层与存储/RAG 层之间，提供：
- extraction_service:  文档解析（PDF/图片 -> 结构化字段）
- generation_service:  内容生成（自我介绍/项目介绍/求职信）
- preparation_service: 面试准备（匹配分析/复习计划/每日回顾）
- tracking_service:    投递跟踪（今日待办/到期检查/状态统计）

所有服务均为无状态单例。
"""
