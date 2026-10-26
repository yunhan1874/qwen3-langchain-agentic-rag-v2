from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

from rag_v2.agent import AgenticRAGEngine, LangChainAgenticRAG
from rag_v2.compression import ExtractiveContextCompressor
from rag_v2.config import RAGConfig
from rag_v2.embeddings import HashingEmbedder
from rag_v2.generation import ExtractiveGenerator
from rag_v2.models import SourceDocument
from rag_v2.query_rewrite import RuleBasedRewriter
from rag_v2.reranker import TokenOverlapReranker
from rag_v2.retrieval import HybridRetriever, InMemoryDenseIndex


@unittest.skipUnless(importlib.util.find_spec("langchain_core"), "langchain-core not installed")
class LangChainLCELTests(unittest.TestCase):
    def test_public_invoke_runs_lcel_graph(self) -> None:
        documents = [
            SourceDocument(
                "hypertension",
                "高血压患者需要监测血压并遵循医生建议。",
                {"source": "docs.txt", "document_type": "medical_qa"},
            )
        ]
        dense = InMemoryDenseIndex(documents, HashingEmbedder())
        retriever = HybridRetriever(documents, dense, fused_k=1)
        config = replace(
            RAGConfig(),
            min_relevance_score=0.2,
            final_k=1,
            rerank_k=1,
            fused_k=1,
        )
        engine = AgenticRAGEngine(
            retriever=retriever,
            reranker=TokenOverlapReranker(),
            rewriter=RuleBasedRewriter(),
            compressor=ExtractiveContextCompressor(100),
            generator=ExtractiveGenerator(),
            config=config,
        )
        rag = LangChainAgenticRAG(engine)
        answer = rag.invoke("请问高血压如何管理？", {"document_type": "medical_qa"})
        mermaid = rag.chain.get_graph().draw_mermaid()
        self.assertEqual("answered", answer.status)
        self.assertIn("prepare_query", mermaid)
        self.assertIn("agentic_hybrid_retrieval", mermaid)
        self.assertIn("cited_grounded_generation", mermaid)


if __name__ == "__main__":
    unittest.main()
