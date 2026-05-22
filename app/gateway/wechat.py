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
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ── 统一消息格式 ──────────────────────────────────


class GatewayMessage(BaseModel):
    """统一网关消息 — 屏蔽不同平台的差异"""

    msg_type: str = "text"
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
    ) -> str | bool:
        """验证微信服务器签名。"""
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
        """解析微信 XML 消息 → GatewayMessage。"""
        try:
            root = ET.fromstring(xml_data)
            msg_type = self._get_text(root, "MsgType", "text")
            content = self._get_text(root, "Content", "")
            from_user = self._get_text(root, "FromUserName", "")
            create_time = int(self._get_text(root, "CreateTime", "0"))

            return GatewayMessage(
                msg_type=msg_type,
                content=content,
                from_user=from_user,
                timestamp=datetime.fromtimestamp(create_time) if create_time else datetime.now(),
                raw={
                    "to_user": self._get_text(root, "ToUserName", ""),
                    "msg_id": self._get_text(root, "MsgId", ""),
                },
            )
        except ET.ParseError as e:
            logger.error("XML 解析失败: %s", e)
            return GatewayMessage(content=xml_data, raw={"parse_error": str(e)})

    # ── 回复构建 ─────────────────────────────────

    def build_reply(self, to_user: str, from_user: str, content: str) -> str:
        """构建回复 XML。"""
        ts = int(time.time())
        return (
            "<xml>"
            f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
            f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
            f"<CreateTime>{ts}</CreateTime>"
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
