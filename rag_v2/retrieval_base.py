from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

from .embeddings import Embedder
from .models import RetrievalHit, SourceDocument
from .text import cosine, tokenize


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class BM25Index:
    def __init__(self, documents: list[SourceDocument], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document.text) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, frequencies in enumerate(self.term_frequencies):
            for term, frequency in frequencies.items():
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / (self.average_length or 1)
                )
                contribution = self.idf[term] * frequency * (self.k1 + 1) / denominator
                self.postings[term].append((index, contribution))

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        allowed = None
        if filters:
            allowed = {
                index for index, document in enumerate(self.documents)
                if metadata_matches(document.metadata, filters)
            }
            if not allowed:
                return []
        scores: dict[int, float] = defaultdict(float)
        # Preserve repeated query terms and their original accumulation order.
        for term in tokenize(query):
            for index, contribution in self.postings.get(term, ()):
                if allowed is None or index in allowed:
                    scores[index] += contribution
        scored = [
            (self.documents[index].document_id, score)
            for index, score in scores.items() if score > 0
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


class DenseIndex(Protocol):
    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]: ...


class InMemoryDenseIndex:
    def __init__(self, documents: list[SourceDocument], embedder: Embedder) -> None:
        self.documents = documents
        self.embedder = embedder
        self.vectors = embedder.embed_documents([document.text for document in documents])

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        query_vector = self.embedder.embed_query(query)
        results = [
            (document.document_id, max(0.0, cosine(query_vector, vector)))
            for document, vector in zip(self.documents, self.vectors, strict=True)
            if metadata_matches(document.metadata, filters)
        ]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:k]


class ChromaDenseIndex:
    """Persistent dense index; metadata filters are pushed down into Chroma."""

    def __init__(
        self,
        documents: list[SourceDocument],
        embedder: Embedder,
        persist_directory: str | Path,
        collection_name: str,
        rebuild: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("请先安装 chromadb") from exc
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(persist_directory))
        if rebuild:
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )
        if rebuild or self.collection.count() == 0:
            self._add_documents(documents)

    def _add_documents(self, documents: list[SourceDocument]) -> None:
        if not documents:
            return
        vectors = self.embedder.embed_documents([document.text for document in documents])
        metadatas = []
        for document in documents:
            metadata = {
                key: value
                for key, value in document.metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            metadatas.append(metadata or {"source": "unknown"})
        self.collection.add(
            ids=[document.document_id for document in documents],
            documents=[document.text for document in documents],
            embeddings=vectors,
            metadatas=metadatas,
        )

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            clauses.append({key: {"$in": list(value)}} if isinstance(value, (list, tuple, set)) else {key: value})
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


class HybridRetriever:
    """Fuse BM25 and dense ranks using RRF; neither route can dominate by scale."""

    def __init__(
        self,
        documents: list[SourceDocument],
        dense_index: DenseIndex,
        dense_k: int = 12,
        bm25_k: int = 12,
        fused_k: int = 12,
        rrf_constant: int = 60,
    ) -> None:
        self.documents = {document.document_id: document for document in documents}
        self.dense = dense_index
        self.bm25 = BM25Index(documents)
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.fused_k = fused_k
        self.rrf_constant = rrf_constant

    def has_matching_documents(self, filters: dict[str, Any]) -> bool:
        return any(metadata_matches(document.metadata, filters) for document in self.documents.values())

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        dense = self.dense.search(query, self.dense_k, filters)
        lexical = self.bm25.search(query, self.bm25_k, filters)
        dense_scores = dict(dense)
        bm25_scores = dict(lexical)
        fused: dict[str, float] = defaultdict(float)
        for rank, (document_id, _) in enumerate(dense, 1):
            fused[document_id] += 1 / (self.rrf_constant + rank)
        for rank, (document_id, _) in enumerate(lexical, 1):
            fused[document_id] += 1 / (self.rrf_constant + rank)
        maximum = 2 / (self.rrf_constant + 1)
        hits = [
            RetrievalHit(
                document=self.documents[document_id],
                dense_score=dense_scores.get(document_id, 0.0),
                bm25_score=bm25_scores.get(document_id, 0.0),
                fusion_score=min(1.0, score / maximum),
            )
            for document_id, score in fused.items()
        ]
        hits.sort(
            key=lambda hit: (
                -hit.fusion_score,
                -hit.dense_score,
                -hit.bm25_score,
                hit.document.document_id,
            )
        )
        return hits[: self.fused_k]
