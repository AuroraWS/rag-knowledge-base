"""LLMGenerator 单元测试"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_string(self, mock_llm_generator):
        """generate() 应返回字符串。"""
        result = await mock_llm_generator.generate("测试提示")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_structured_returns_dict(self, mock_llm_generator):
        """generate_structured() 应返回 dict。"""
        with patch.object(mock_llm_generator, '_call_api') as mock_call:
            mock_call.return_value = {
                "choices": [{"message": {"content": '{"key": "value"}'}}]
            }
            result = await mock_llm_generator.generate_structured("生成JSON")
            assert isinstance(result, dict)
            assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_empty_api_key_raises(self):
        """空 API key 应引发 ValueError。"""
        from app.rag.generator import LLMGenerator
        gen = LLMGenerator(api_key="")
        with patch.object(gen, '_ensure_api_key') as mock_ensure:
            mock_ensure.side_effect = ValueError("DEEPSEEK_API_KEY 未设置")
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                await gen.generate("测试")

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_llm_generator):
        """异步上下文管理器应正常工作。"""
        async with mock_llm_generator as gen:
            assert gen is mock_llm_generator

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_llm_generator):
        """有 system_prompt 时应传入 messages[0] 为 system role。"""
        with patch.object(mock_llm_generator, '_call_api') as mock_call:
            mock_call.return_value = {
                "choices": [{"message": {"content": "回答"}}]
            }
            await mock_llm_generator.generate("问题", system_prompt="你是一个助手")
            messages = mock_call.call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "你是一个助手"
