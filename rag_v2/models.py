from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    document: SourceDocument
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def score(self) -> float:
        return self.rerank_score or self.fusion_score


@dataclass(frozen=True)
class Citation:
    citation_id: int
    chunk_id: str
    source: str
    title: str
    excerpt: str
    evidence: str = ""


@dataclass(frozen=True)
class RetrievalDetail:
    chunk_id: str
    source: str
    title: str
    excerpt: str
    dense_score: float
    bm25_score: float
    fusion_score: float
    rerank_score: float
    selected: bool


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    original_query: str
    rewritten_query: str
    status: str
    agent_steps: list[str]
    retrieved_ids: list[str]
    confidence: float
    retrieval_details: list[RetrievalDetail] = field(default_factory=list)
    citation_status: str = "not_applicable"

    def render(self) -> str:
        if not self.citations:
            return self.answer
        sources = ["", "检索证据（不代表逐句事实核验）："]
        for citation in self.citations:
            label = citation.title or citation.source
            sources.append(f"[{citation.citation_id}] {label} — {citation.excerpt}")
        return "\n".join([self.answer, *sources])
