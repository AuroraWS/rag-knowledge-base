"""内容生成服务 — 自我介绍 / 项目介绍 / 求职信生成。

使用 LLMGenerator 基于用户简历信息和岗位描述生成定制化内容。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

# ── 系统提示词 ──────────────────────────────────────

SELF_INTRO_SYSTEM_PROMPT = """你是一位专业的招聘文案撰写专家。请根据用户的简历信息和目标岗位描述(JD)，生成高质量的自我介绍。

输出必须是一个 JSON 对象（不要包含 markdown 代码块标记），包含以下字段：

{
    "short": "50字以内的精简版自我介绍，适合在线简历摘要或社交平台简介",
    "full": "200-300字的详细版自我介绍，适合求职信开头或面试开场，突出与JD的匹配点",
    "english": "英文版自我介绍（100-150词），适合外企或英文面试"
}

注意：
- 使用第一人称
- 突出与目标岗位匹配的技能和经验
- 语气专业、自信但不浮夸
"""

PROJECT_INTRO_SYSTEM_PROMPT = """你是一位专业的面试辅导专家。请将用户的项目经历转化为 STAR 格式的面试回答。

STAR 格式说明：
- Situation (情境): 项目背景、业务场景
- Task (任务): 你在项目中承担的任务和目标
- Action (行动): 你采取的具体行动和技术方案
- Result (结果): 取得的成果和数据指标

根据用户指定的长度输出：
- "short" (标准): 150-200字，重点突出
- "detailed" (详细): 300-400字，包含更多技术细节
"""

COVER_LETTER_SYSTEM_PROMPT = """你是一位专业的求职信撰写专家。请根据用户的简历信息和目标岗位描述(JD)，生成一封正式的中文求职信（自荐信）。

要求：
- 称呼得体（尊敬的XX公司招聘负责人）
- 开头表达对公司和岗位的兴趣
- 中间段落突出与岗位匹配的技能、经验和成就
- 结尾表达期待面试机会
- 署名：用户姓名
- 整体语气：专业、诚恳、自信
- 字数：300-500字
- 不要使用占位符，所有信息基于用户真实简历
"""


class GenerationService:
    """内容生成服务 — 生成自我介绍、项目介绍、求职信等。"""

    def __init__(self, generator: Optional[LLMGenerator] = None) -> None:
        self._generator = generator or LLMGenerator()

    # ── 自我介绍生成 ────────────────────────────────

    async def generate_self_intro(
        self,
        jd_text: str,
        profile_data: dict[str, Any],
    ) -> dict[str, str]:
        """根据 JD 和简历数据生成三个版本的自我介绍。

        Args:
            jd_text: 目标岗位描述全文。
            profile_data: 简历数据字典，包含 personal_info, education,
                          work_experience, projects 等字段。

        Returns:
            {"short": "...", "full": "...", "english": "..."}
        """
        profile_text = self._format_profile(profile_data)

        prompt = f"""请根据以下简历信息和目标岗位，生成自我介绍。

===== 简历信息 =====
{profile_text}

===== 目标岗位描述(JD) =====
{jd_text}

请生成短版、详细版和英文版三个版本的自我介绍。"""

        try:
            result = await self._generator.generate_structured(
                prompt=prompt,
                system_prompt=SELF_INTRO_SYSTEM_PROMPT,
                temperature=0.7,
            )

            if isinstance(result, str):
                result = json.loads(result)

            return {
                "short": result.get("short", ""),
                "full": result.get("full", ""),
                "english": result.get("english", ""),
            }

        except Exception as e:
            logger.error("自我介绍生成失败: %s", e, exc_info=True)
            return {
                "short": "生成失败，请稍后重试",
                "full": "生成失败，请稍后重试",
                "english": "Generation failed, please try again later",
            }

    # ── 项目介绍生成 ────────────────────────────────

    async def generate_project_intro(
        self,
        project_data: dict[str, Any],
        length: str = "standard",
    ) -> str:
        """将项目经历转化为 STAR 格式的面试回答。

        Args:
            project_data: 项目数据字典，包含 name, role, description,
                          tech_stack, highlights 等字段。
            length: "standard"（标准）或 "detailed"（详细）。

        Returns:
            STAR 格式的项目介绍文本。
        """
        project_text = json.dumps(project_data, ensure_ascii=False, indent=2)

        prompt = f"""请将以下项目经历转化为 STAR 格式的面试回答（{length} 长度）。

===== 项目信息 =====
{project_text}

请输出 STAR 格式的回答。"""

        try:
            result = await self._generator.generate(
                prompt=prompt,
                system_prompt=PROJECT_INTRO_SYSTEM_PROMPT,
                temperature=0.7,
            )
            return result

        except Exception as e:
            logger.error("项目介绍生成失败: %s", e, exc_info=True)
            return "生成失败，请稍后重试"

    # ── 求职信生成 ──────────────────────────────────

    async def generate_cover_letter(
        self,
        company: str,
        position: str,
        jd_text: str,
        profile_data: dict[str, Any],
    ) -> str:
        """生成针对特定公司和岗位的求职信。

        Args:
            company: 公司名称。
            position: 岗位名称。
            jd_text: 岗位描述全文。
            profile_data: 简历数据字典。

        Returns:
            求职信文本。
        """
        profile_text = self._format_profile(profile_data)
        user_name = self._get_name(profile_data)

        prompt = f"""请为 {user_name} 撰写一封给 {company} 的 {position} 岗位求职信。

===== 简历信息 =====
{profile_text}

===== 目标公司 =====
{company}

===== 目标岗位 =====
{position}

===== 岗位描述(JD) =====
{jd_text}

请生成一封正式、专业的中文求职信。"""

        try:
            result = await self._generator.generate(
                prompt=prompt,
                system_prompt=COVER_LETTER_SYSTEM_PROMPT,
                temperature=0.7,
            )
            return result

        except Exception as e:
            logger.error("求职信生成失败: %s", e, exc_info=True)
            return "生成失败，请稍后重试"

    # ── 辅助方法 ────────────────────────────────────

    @staticmethod
    def _format_profile(profile: dict[str, Any]) -> str:
        """将简历数据格式化为适合 LLM 阅读的文本。"""
        parts = []

        # 个人信息
        pi = profile.get("personal_info")
        if pi:
            name = getattr(pi, "name", pi.get("name", "")) if isinstance(pi, object) else pi.get("name", "")
            phone = getattr(pi, "phone", pi.get("phone", "")) if isinstance(pi, object) else pi.get("phone", "")
            email = getattr(pi, "email", pi.get("email", "")) if isinstance(pi, object) else pi.get("email", "")
            parts.append(f"姓名: {name}")
            parts.append(f"电话: {phone}")
            parts.append(f"邮箱: {email}")

        # 教育经历
        edu_list = profile.get("education", [])
        if edu_list:
            parts.append("\n【教育经历】")
            for e in edu_list:
                school = getattr(e, "school", e.get("school", "")) if isinstance(e, object) else e.get("school", "")
                degree = getattr(e, "degree", e.get("degree", "")) if isinstance(e, object) else e.get("degree", "")
                major = getattr(e, "major", e.get("major", "")) if isinstance(e, object) else e.get("major", "")
                parts.append(f"- {school} | {degree} | {major}")

        # 工作经历
        work_list = profile.get("work_experience", [])
        if work_list:
            parts.append("\n【工作经历】")
            for w in work_list:
                company = getattr(w, "company", w.get("company", "")) if isinstance(w, object) else w.get("company", "")
                title = getattr(w, "title", w.get("title", "")) if isinstance(w, object) else w.get("title", "")
                parts.append(f"- {company} | {title}")

        # 项目经历
        proj_list = profile.get("projects", [])
        if proj_list:
            parts.append("\n【项目经历】")
            for p in proj_list:
                name = getattr(p, "name", p.get("name", "")) if isinstance(p, object) else p.get("name", "")
                role = getattr(p, "role", p.get("role", "")) if isinstance(p, object) else p.get("role", "")
                parts.append(f"- {name} | {role}")

        return "\n".join(parts)

    @staticmethod
    def _get_name(profile: dict[str, Any]) -> str:
        """从 profile 数据中提取用户姓名。"""
        pi = profile.get("personal_info")
        if pi:
            if hasattr(pi, "name"):
                return pi.name
            if isinstance(pi, dict):
                return pi.get("name", "用户")
        return "用户"

    async def close(self) -> None:
        """释放资源。"""
        await self._generator.close()


# 模块级单例
generation_service = GenerationService()
