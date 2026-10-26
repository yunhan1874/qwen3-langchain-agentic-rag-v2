from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from rag_v2.embeddings import HashingEmbedder
from rag_v2.models import SourceDocument
from rag_v2.retrieval import ChromaDenseIndex


class FakeCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.batch_sizes: list[int] = []

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids, documents, embeddings, metadatas) -> None:
        self.batch_sizes.append(len(ids))
        self.ids.extend(ids)
        self.last_payload = (documents, embeddings, metadatas)


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_or_create_collection(self, name, configuration):
        self.last_create = (name, configuration)
        return self.collection

    def create_collection(self, name, configuration):
        self.collection = FakeCollection()
        self.last_create = (name, configuration)
        return self.collection

    def delete_collection(self, name) -> None:
        del name
        self.collection = FakeCollection()


class ChromaBatchingTests(unittest.TestCase):
    def test_large_corpus_is_written_in_bounded_batches(self) -> None:
        client = FakeClient()
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        documents = [
            SourceDocument(str(index), f"高血压资料 {index}", {"source": "docs.txt"})
            for index in range(600)
        ]
        with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
            index = ChromaDenseIndex(
                documents,
                HashingEmbedder(32),
                "/tmp/not-used-by-fake",
                "medical",
                rebuild=True,
                batch_size=256,
            )
        self.assertEqual([256, 256, 88], index.collection.batch_sizes)
        self.assertEqual(600, index.collection.count())


if __name__ == "__main__":
    unittest.main()
