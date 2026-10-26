from __future__ import annotations

from .models import RetrievalHit, SourceDocument
from .text import split_sentences, tokenize


class ExtractiveContextCompressor:
    """Keep query-relevant sentences while preserving chunk identity and citation metadata."""

    def __init__(self, max_chars_per_document: int = 650) -> None:
        self.max_chars_per_document = max_chars_per_document

    def compress(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        query_tokens = set(tokenize(query))
        compressed: list[RetrievalHit] = []
        for hit in hits:
            sentences = split_sentences(hit.document.text) or [hit.document.text]
            ranked = sorted(
                enumerate(sentences),
                key=lambda item: (
                    -len(query_tokens & set(tokenize(item[1]))),
                    item[0],
                ),
            )
            selected: list[tuple[int, str]] = []
            total = 0
            for index, sentence in ranked:
                remaining = self.max_chars_per_document - total
                if remaining <= 0:
                    break
                if len(sentence) > remaining:
                    if not selected:
                        selected.append((index, sentence[:remaining]))
                        total += remaining
                    continue
                selected.append((index, sentence))
                total += len(sentence)
            selected.sort(key=lambda item: item[0])
            text = "".join(sentence for _, sentence in selected).strip()
            if not text:
                continue
            metadata = {**hit.document.metadata, "compressed": True}
            compressed.append(
                RetrievalHit(
                    document=SourceDocument(hit.document.document_id, text, metadata),
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    fusion_score=hit.fusion_score,
                    rerank_score=hit.rerank_score,
                )
            )
        return compressed
