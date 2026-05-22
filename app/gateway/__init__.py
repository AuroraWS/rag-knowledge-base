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
