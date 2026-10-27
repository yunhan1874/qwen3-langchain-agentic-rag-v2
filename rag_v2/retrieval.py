from __future__ import annotations

from pathlib import Path
from typing import Any

from .embeddings import Embedder
from .models import SourceDocument
from .retrieval_base import (
    BM25Index,
    DenseIndex,
    HybridRetriever,
    InMemoryDenseIndex,
    metadata_matches,
)


class ChromaDenseIndex:
    """Persistent batched dense index with metadata filter pushdown."""

    def __init__(
        self,
        documents: list[SourceDocument],
        embedder: Embedder,
        persist_directory: str | Path,
        collection_name: str,
        rebuild: bool = False,
        batch_size: int = 256,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("请先安装 chromadb") from exc
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )
        existing_count = self.collection.count()
        if rebuild and existing_count:
            client.delete_collection(collection_name)
            self.collection = client.create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
            existing_count = 0
        if existing_count not in {0, len(documents)}:
            raise RuntimeError("Chroma 内容与 chunks 不一致；请使用新的索引版本或显式重建")
        if existing_count == 0:
            self._add_documents(documents, batch_size)
        else:
            self._validate_documents(documents, batch_size)
        if self.collection.count() != len(documents):
            raise RuntimeError("Chroma 向量数量与 semantic chunks 数量不一致，请重建索引")

    @staticmethod
    def _metadata(document: SourceDocument) -> dict[str, Any]:
        return {
            key: value for key, value in document.metadata.items()
            if isinstance(value, (str, int, float, bool))
        } or {"source": "unknown"}

    def _validate_documents(self, documents: list[SourceDocument], batch_size: int) -> None:
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            stored = self.collection.get(ids=[d.document_id for d in batch], include=["documents", "metadatas"])
            rows = {
                key: (text, metadata) for key, text, metadata in zip(
                    stored["ids"], stored["documents"], stored["metadatas"], strict=True
                )
            }
            if any(rows.get(d.document_id) != (d.text, self._metadata(d)) for d in batch):
                raise RuntimeError("Chroma ID、文本或元信息已变化；请使用新的索引版本或显式重建")

    def _add_documents(self, documents: list[SourceDocument], batch_size: int) -> None:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            vectors = self.embedder.embed_documents([document.text for document in batch])
            metadatas = [self._metadata(document) for document in batch]
            self.collection.add(
                ids=[document.document_id for document in batch],
                documents=[document.text for document in batch],
                embeddings=vectors,
                metadatas=metadatas,
            )

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                clauses.append({key: {"$in": list(value)}})
            else:
                clauses.append({key: value})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, count),
            where=self._where(filters),
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            (document_id, max(0.0, min(1.0, 1.0 - float(distance))))
            for document_id, distance in zip(ids, distances, strict=False)
        ]


__all__ = [
    "BM25Index",
    "ChromaDenseIndex",
    "DenseIndex",
    "HybridRetriever",
    "InMemoryDenseIndex",
    "metadata_matches",
]
