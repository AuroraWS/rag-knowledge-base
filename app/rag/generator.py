"""
LLM 生成模块 — 基于 httpx 调用 DeepSeek API（OpenAI 兼容）

支持:
- generate:          单轮文本生成
- generate_chat:     多轮对话
- generate_structured: JSON 模式（结构化输出）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Union

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMGenerator:
    """DeepSeek API 调用封装（OpenAI 兼容接口，httpx 实现）"""

    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TEMPERATURE = 0.7
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # 秒

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self._api_key = api_key or settings.deepseek_api_key
        self._base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self._model = model or self.DEFAULT_MODEL
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── HTTP 客户端管理 ────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
        return self._client

    async def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中配置或设置环境变量。"
            )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── 核心 API 调用 ──────────────────────────────────

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        调用 DeepSeek Chat API。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式的消息列表
        temperature : float
            生成温度
        max_tokens : int
            最大输出 token 数
        response_format : dict, optional
            如 {"type": "json_object"} 启用 JSON 模式

        Returns
        -------
        dict
            API 返回的完整响应（解析后的 JSON）
        """
        await self._ensure_api_key()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        client = await self._get_client()

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "API 调用超时 (attempt %d/%d): %s", attempt, self.MAX_RETRIES, e
                )
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:500]
                logger.error(
                    "API 返回错误 %d (attempt %d/%d): %s",
                    status,
                    attempt,
                    self.MAX_RETRIES,
                    body,
                )
                last_error = e
                # 4xx 错误不重试
                if 400 <= status < 500:
                    raise
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    "网络请求失败 (attempt %d/%d): %s", attempt, self.MAX_RETRIES, e
                )

            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAY * (2 ** (attempt - 1))
                logger.info("等待 %.1f 秒后重试...", delay)
                time.sleep(delay)  # 同步 sleep 在 async 中可被接受（短时间）

        raise RuntimeError(
            f"API 调用在 {self.MAX_RETRIES} 次尝试后全部失败: {last_error}"
        )

    # ── 公共方法 ───────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        单轮文本生成。

        Parameters
        ----------
        prompt : str
            用户提示
        system_prompt : str
            系统提示词，可选
        temperature : float
            生成温度
        max_tokens : int
            最大输出 token 数

        Returns
        -------
        str
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = await self._call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return data["choices"][0]["message"]["content"].strip()

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        多轮对话生成。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式的消息列表
        temperature : float
            生成温度

        Returns
        -------
        str
            生成的回复文本
        """
        data = await self._call_api(
            messages=messages,
            temperature=temperature,
        )
        return data["choices"][0]["message"]["content"].strip()

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        response_model: Optional[type] = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        生成结构化输出（JSON 模式）。

        Parameters
        ----------
        prompt : str
            用户提示
        system_prompt : str
            系统提示词，可选
        response_model : type, optional
            Pydantic BaseModel 类，用于定义输出结构。如果传入，会尝试解析响应。
        temperature : float
            生成温度（结构化输出推荐更低温度）

        Returns
        -------
        dict
            解析后的 JSON 字典。如果提供了 response_model，则是 BaseModel 实例。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = await self._call_api(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        content = data["choices"][0]["message"]["content"].strip()
        result = json.loads(content)

        if response_model is not None:
            return response_model.model_validate(result)

        return result

    async def __aenter__(self) -> "LLMGenerator":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"<LLMGenerator model={self._model} base_url={self._base_url}>"
