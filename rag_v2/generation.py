from __future__ import annotations

from typing import Any, Protocol

from .text import strip_thinking


class AnswerGenerator(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class LangChainGroundedGenerator:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @staticmethod
    def _text(response: Any) -> str:
        return response.content if hasattr(response, "content") else str(response)

    def generate(self, query: str, context: str) -> str:
        prompt = f"""你是医学专家。只能依据给定证据回答，不得使用证据之外的事实。
每个医学事实后必须标注对应来源编号，例如 [1]。如果证据不足，明确说无法从知识库确认。
不要输出思考过程。

证据：
{context}

问题：{query}
回答："""
        return strip_thinking(self._text(self.llm.invoke(prompt)))


class ExtractiveGenerator:
    """Test double that produces a grounded, cited answer without a model."""

    def generate(self, query: str, context: str) -> str:
        del query
        first = context.splitlines()
        evidence = next((line for line in first if line and not line.startswith("[")), "证据不足")
        return f"{evidence} [1]"
