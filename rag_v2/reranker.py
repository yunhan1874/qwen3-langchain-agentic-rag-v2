from __future__ import annotations

import math
from typing import Protocol

from .models import RetrievalHit
from .text import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]: ...


def _normalise_score(value: float) -> float:
    if 0 <= value <= 1:
        return value
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("请先安装 sentence-transformers") from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        raw_scores = self.model.predict([(query, hit.document.text) for hit in hits])
        reranked = [
            RetrievalHit(
                document=hit.document,
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                fusion_score=hit.fusion_score,
                rerank_score=0.85 * _normalise_score(float(score)) + 0.15 * hit.fusion_score,
            )
            for hit, score in zip(hits, raw_scores, strict=True)
        ]
        reranked.sort(key=lambda hit: (-hit.rerank_score, hit.document.document_id))
        return reranked[:top_k]


class TokenOverlapReranker:
    """Dependency-free reranker for unit tests; not the production default."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        query_tokens = set(tokenize(query))
        reranked = []
        for hit in hits:
            document_tokens = set(tokenize(hit.document.text))
            overlap = len(query_tokens & document_tokens) / (len(query_tokens) or 1)
            reranked.append(
                RetrievalHit(
                    document=hit.document,
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    fusion_score=hit.fusion_score,
                    rerank_score=0.85 * overlap + 0.15 * hit.fusion_score,
                )
            )
        reranked.sort(key=lambda hit: (-hit.rerank_score, hit.document.document_id))
        return reranked[:top_k]
