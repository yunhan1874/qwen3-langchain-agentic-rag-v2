from __future__ import annotations

import re

from .models import Citation, RetrievalHit
from .text import strip_thinking


_CITATION = re.compile(r"\[(\d+)\]")


def build_citations(hits: list[RetrievalHit]) -> tuple[str, list[Citation]]:
    context_parts: list[str] = []
    citations: list[Citation] = []
    for number, hit in enumerate(hits, 1):
        metadata = hit.document.metadata
        source = str(metadata.get("source", "未知来源"))
        title = str(metadata.get("title", ""))
        evidence = strip_thinking(hit.document.text)
        excerpt = evidence[:180].replace("\n", " ")
        citations.append(
            Citation(
                citation_id=number,
                chunk_id=hit.document.document_id,
                source=source,
                title=title,
                excerpt=excerpt,
                evidence=evidence,
            )
        )
        context_parts.append(
            f"[{number}] source={source}; chunk_id={hit.document.document_id}\n"
            f"{evidence}"
        )
    return "\n\n".join(context_parts), citations


def enforce_valid_citations(answer: str, citations: list[Citation]) -> str:
    valid = {citation.citation_id for citation in citations}

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if number in valid else ""

    cleaned = _CITATION.sub(replace, answer).strip()
    return cleaned


def has_valid_citation(answer: str, citations: list[Citation]) -> bool:
    valid = {citation.citation_id for citation in citations}
    return any(int(number) in valid for number in _CITATION.findall(answer))
