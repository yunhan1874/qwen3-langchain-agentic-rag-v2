from __future__ import annotations

import re

from .models import Citation, RetrievalHit


_CITATION = re.compile(r"\[(\d+)\]")


def build_citations(hits: list[RetrievalHit]) -> tuple[str, list[Citation]]:
    context_parts: list[str] = []
    citations: list[Citation] = []
    for number, hit in enumerate(hits, 1):
        metadata = hit.document.metadata
        source = str(metadata.get("source", "未知来源"))
        title = str(metadata.get("title", ""))
        excerpt = hit.document.text[:180].replace("\n", " ")
        citations.append(
            Citation(
                citation_id=number,
                chunk_id=hit.document.document_id,
                source=source,
                title=title,
                excerpt=excerpt,
            )
        )
        context_parts.append(
            f"[{number}] source={source}; chunk_id={hit.document.document_id}\n"
            f"{hit.document.text}"
        )
    return "\n\n".join(context_parts), citations


def enforce_valid_citations(answer: str, citations: list[Citation]) -> str:
    valid = {citation.citation_id for citation in citations}

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if number in valid else ""

    cleaned = _CITATION.sub(replace, answer).strip()
    used = {int(number) for number in _CITATION.findall(cleaned)}
    if citations and not used:
        cleaned = f"{cleaned} [{citations[0].citation_id}]".strip()
    return cleaned
