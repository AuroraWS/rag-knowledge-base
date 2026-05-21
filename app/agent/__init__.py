"""智能招聘助手 — Agent 模块

基于 LangGraph 的工作流编排 + APScheduler 定时调度。

包含：
- workflow: LangGraph StateGraph 工作流定义
  - 投递跟踪工作流（Application Tracking Workflow）
  - 面试准备流程（Interview Prep Workflow）
- scheduler: APScheduler 定时任务
  - 每日回顾推送（每天 09:30）
  - 每日日志生成（每天 22:00）
"""
