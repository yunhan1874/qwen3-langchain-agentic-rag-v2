from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .text import strip_thinking


ALLOWED_FILTERS = {
    "source",
    "source_line",
    "document_type",
    "department",
    "category",
    "year",
}


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    rewritten_query: str
    filters: dict[str, Any]


class Rewriter(Protocol):
    def rewrite(
        self,
        query: str,
        explicit_filters: dict[str, Any] | None = None,
    ) -> QueryPlan: ...

    def retry_query(self, plan: QueryPlan, attempt: int, feedback: str) -> str: ...


class RuleBasedRewriter:
    """Deterministic fallback used for offline tests and graceful degradation."""

    _PREFIXES = (
        "请问",
        "我想知道",
        "麻烦介绍一下",
        "能不能告诉我",
        "医生您好",
    )

    def rewrite(
        self,
        query: str,
        explicit_filters: dict[str, Any] | None = None,
    ) -> QueryPlan:
        rewritten = query.strip()
        for prefix in self._PREFIXES:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip("，,：: ")
                break
        rewritten = re.sub(r"\s+", " ", rewritten)
        return QueryPlan(query.strip(), rewritten or query.strip(), explicit_filters or {})

    def retry_query(self, plan: QueryPlan, attempt: int, feedback: str) -> str:
        del feedback
        if attempt == 1 and plan.rewritten_query != plan.original_query:
            return plan.original_query
        return re.sub(r"[？?，,。.!！]", " ", plan.rewritten_query).strip()


class LLMQueryRewriter:
    """Use the existing Qwen LangChain LLM to rewrite and extract explicit metadata filters."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.fallback = RuleBasedRewriter()

    @staticmethod
    def _text(response: Any) -> str:
        text = response.content if hasattr(response, "content") else str(response)
        return strip_thinking(text)

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("query rewrite did not return JSON")
        return json.loads(match.group(0))

    def rewrite(
        self,
        query: str,
        explicit_filters: dict[str, Any] | None = None,
    ) -> QueryPlan:
        prompt = f"""你是医学知识库检索查询改写器。只改写检索词，不回答问题。
要求：补全指代和医学同义词；保留原意；只有用户明确给出来源、科室、类别或年份时才生成过滤条件。
只输出 JSON：{{"query":"...","filters":{{}}}}
允许的过滤字段：{sorted(ALLOWED_FILTERS)}
原始问题：{query}
"""
        try:
            payload = self._json_object(self._text(self.llm.invoke(prompt)))
            rewritten = str(payload.get("query", "")).strip() or query.strip()
            inferred = {
                key: value
                for key, value in dict(payload.get("filters") or {}).items()
                if key in ALLOWED_FILTERS
            }
        except (ValueError, TypeError, json.JSONDecodeError):
            return self.fallback.rewrite(query, explicit_filters)
        merged = {**inferred, **(explicit_filters or {})}
        return QueryPlan(query.strip(), rewritten, merged)

    def retry_query(self, plan: QueryPlan, attempt: int, feedback: str) -> str:
        prompt = f"""你是检索 Agent。上次查询召回不足，请生成一个更适合医学知识库的查询。
不得回答问题，不得添加原问题没有的疾病事实，只输出一行查询。
原问题：{plan.original_query}
上次查询：{plan.rewritten_query}
第 {attempt + 1} 次尝试；反馈：{feedback}
"""
        try:
            candidate = self._text(self.llm.invoke(prompt)).strip().splitlines()[0]
            return candidate[:300] or self.fallback.retry_query(plan, attempt, feedback)
        except Exception:
            return self.fallback.retry_query(plan, attempt, feedback)
