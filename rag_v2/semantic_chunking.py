from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .embeddings import Embedder
from .models import SourceDocument
from .text import cosine, percentile, split_sentences, stable_id, strip_thinking


def load_source_documents(path: str | Path) -> list[SourceDocument]:
    """Reconstruct docs.py records and attach filterable source metadata."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"文档文件不存在：{source_path}")

    records: list[tuple[int, str]] = []
    current: list[str] = []
    start_line = 1
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("问题:") and current:
                records.append((start_line, "\n".join(current)))
                current = []
            if not current:
                start_line = line_number
            current.append(line)
    if current:
        records.append((start_line, "\n".join(current)))

    documents: list[SourceDocument] = []
    seen: set[str] = set()
    for line_number, text in records:
        text = strip_thinking(text)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        document_id = stable_id(str(source_path.resolve()), str(line_number), text)
        first_line = text.splitlines()[0]
        documents.append(
            SourceDocument(
                document_id=document_id,
                text=text,
                metadata={
                    "source": source_path.name,
                    "source_path": str(source_path.resolve()),
                    "source_line": line_number,
                    "title": first_line[:120],
                    "document_type": "medical_qa",
                },
            )
        )
    return documents


class SemanticChunker:
    """Split at low adjacent-sentence semantic similarity, with size safety bounds."""

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 25.0,
        min_chunk_chars: int = 120,
        max_chunk_chars: int = 900,
    ) -> None:
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def split_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        chunks: list[SourceDocument] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: SourceDocument) -> list[SourceDocument]:
        sentences = split_sentences(document.text) or [document.text]
        if len(sentences) == 1:
            return [self._make_chunk(document, sentences, 0, 0, 0)]

        vectors = self.embedder.embed_documents(sentences)
        similarities = [
            cosine(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]
        breakpoint = percentile(similarities, self.breakpoint_percentile)

        groups: list[tuple[list[str], int, int]] = []
        current = [sentences[0]]
        start = 0
        for index, similarity in enumerate(similarities):
            following = sentences[index + 1]
            current_chars = sum(len(sentence) for sentence in current)
            projected_chars = current_chars + len(following)
            semantic_break = similarity <= breakpoint and current_chars >= self.min_chunk_chars
            forced_break = projected_chars > self.max_chunk_chars and bool(current)
            if semantic_break or forced_break:
                groups.append((current, start, index))
                current = [following]
                start = index + 1
            else:
                current.append(following)
        groups.append((current, start, len(sentences) - 1))

        return [
            self._make_chunk(document, group, chunk_index, begin, end)
            for chunk_index, (group, begin, end) in enumerate(groups)
        ]

    @staticmethod
    def _make_chunk(
        document: SourceDocument,
        sentences: list[str],
        chunk_index: int,
        start_sentence: int,
        end_sentence: int,
    ) -> SourceDocument:
        text = "".join(sentences)
        chunk_id = stable_id(document.document_id, str(chunk_index), text)
        metadata: dict[str, Any] = {
            **document.metadata,
            "parent_document_id": document.document_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "start_sentence": start_sentence,
            "end_sentence": end_sentence,
        }
        return SourceDocument(document_id=chunk_id, text=text, metadata=metadata)


def save_chunks(chunks: list[SourceDocument], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks(path: str | Path) -> list[SourceDocument]:
    chunks: list[SourceDocument] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks.append(
                SourceDocument(
                    document_id=row["document_id"],
                    text=row["text"],
                    metadata=row.get("metadata", {}),
                )
            )
    return chunks
