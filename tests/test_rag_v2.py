from __future__ import annotations

import unittest
from dataclasses import replace

from rag_v2.agent import AgenticRAGEngine
from rag_v2.citations import build_citations, enforce_valid_citations
from rag_v2.compression import ExtractiveContextCompressor
from rag_v2.config import RAGConfig
from rag_v2.embeddings import HashingEmbedder
from rag_v2.evaluation import faithfulness_score, recall_at_k
from rag_v2.generation import ExtractiveGenerator, LangChainGroundedGenerator
from rag_v2.models import RetrievalHit, SourceDocument
from rag_v2.query_rewrite import QueryPlan, RuleBasedRewriter
from rag_v2.reranker import TokenOverlapReranker
from rag_v2.retrieval import HybridRetriever, InMemoryDenseIndex
from rag_v2.semantic_chunking import SemanticChunker
from rag_v2.text import strip_thinking


class FixedEmbedder:
    vectors = {
        "高血压需要监测。": [1.0, 0.0],
        "降压治疗应遵医嘱。": [1.0, 0.0],
        "火箭依靠推力升空。": [0.0, 1.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


class RetryRewriter(RuleBasedRewriter):
    def rewrite(self, query: str, explicit_filters=None) -> QueryPlan:
        return QueryPlan(query, "第一次无结果", explicit_filters or {})

    def retry_query(self, plan: QueryPlan, attempt: int, feedback: str) -> str:
        del plan, attempt, feedback
        return "高血压治疗"


class RetryRetriever:
    def __init__(self, hit: RetrievalHit) -> None:
        self.hit = hit
        self.queries: list[str] = []

    def retrieve(self, query: str, filters=None) -> list[RetrievalHit]:
        del filters
        self.queries.append(query)
        return [self.hit] if query == "高血压治疗" else []


class ThinkingLLM:
    def invoke(self, prompt: str) -> str:
        del prompt
        return "<think>内部推理不应公开。</think>需要监测血压。[1]"


class RAGV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            SourceDocument(
                "cardio-1",
                "高血压治疗需要监测血压并遵循医生建议。",
                {"source": "docs.txt", "department": "心内科"},
            ),
            SourceDocument(
                "endo-1",
                "糖尿病患者需要监测血糖。",
                {"source": "docs.txt", "department": "内分泌科"},
            ),
            SourceDocument(
                "cardio-2",
                "ACEI 是常见的降压药物类别。",
                {"source": "guideline.txt", "department": "心内科"},
            ),
        ]

    def test_semantic_chunking_breaks_on_topic_shift(self) -> None:
        document = SourceDocument(
            "doc",
            "高血压需要监测。降压治疗应遵医嘱。火箭依靠推力升空。",
            {"source": "mixed.txt"},
        )
        chunks = SemanticChunker(
            FixedEmbedder(), breakpoint_percentile=25, min_chunk_chars=1, max_chunk_chars=100
        ).split_document(document)
        self.assertEqual(2, len(chunks))
        self.assertIn("降压治疗", chunks[0].text)
        self.assertEqual("火箭依靠推力升空。", chunks[1].text)
        self.assertEqual("doc", chunks[0].metadata["parent_document_id"])

    def test_hybrid_retrieval_contains_dense_and_bm25_candidates(self) -> None:
        dense = InMemoryDenseIndex(self.documents, HashingEmbedder())
        retriever = HybridRetriever(self.documents, dense, dense_k=2, bm25_k=3, fused_k=3)
        hits = retriever.retrieve("高血压 ACEI 降压")
        self.assertTrue(any(hit.dense_score > 0 for hit in hits))
        self.assertTrue(any(hit.bm25_score > 0 for hit in hits))
        self.assertEqual({"cardio-1", "cardio-2"}, {hit.document.document_id for hit in hits[:2]})

    def test_metadata_filter_is_applied_to_both_routes(self) -> None:
        dense = InMemoryDenseIndex(self.documents, HashingEmbedder())
        retriever = HybridRetriever(self.documents, dense, fused_k=3)
        hits = retriever.retrieve("监测", {"department": "内分泌科"})
        self.assertEqual(["endo-1"], [hit.document.document_id for hit in hits])

    def test_reranker_and_compression(self) -> None:
        hits = [
            RetrievalHit(self.documents[1], fusion_score=0.9),
            RetrievalHit(self.documents[0], fusion_score=0.5),
        ]
        reranked = TokenOverlapReranker().rerank("高血压如何治疗", hits, 2)
        self.assertEqual("cardio-1", reranked[0].document.document_id)
        compressed = ExtractiveContextCompressor(18).compress("监测血压", reranked[:1])
        self.assertLessEqual(len(compressed[0].document.text), 18)
        self.assertTrue(compressed[0].document.metadata["compressed"])

    def test_citations_are_structured_and_invalid_ids_removed(self) -> None:
        hit = RetrievalHit(self.documents[0], rerank_score=0.9)
        context, citations = build_citations([hit])
        self.assertIn("[1]", context)
        answer = enforce_valid_citations("需要监测血压 [9]", citations)
        self.assertNotIn("[9]", answer)
        self.assertTrue(answer.endswith("[1]"))

    def test_agent_retries_then_generates_cited_answer(self) -> None:
        hit = RetrievalHit(self.documents[0], fusion_score=0.8)
        retriever = RetryRetriever(hit)
        config = replace(RAGConfig(), min_relevance_score=0.2, final_k=1, rerank_k=1)
        engine = AgenticRAGEngine(
            retriever=retriever,  # type: ignore[arg-type]
            reranker=TokenOverlapReranker(),
            rewriter=RetryRewriter(),
            compressor=ExtractiveContextCompressor(100),
            generator=ExtractiveGenerator(),
            config=config,
        )
        answer = engine.run("请问怎么控制高血压")
        self.assertEqual("answered", answer.status)
        self.assertEqual(["请问怎么控制高血压", "第一次无结果", "高血压治疗"], retriever.queries)
        self.assertIn("[1]", answer.answer)
        self.assertTrue(any("retry_query" in step for step in answer.agent_steps))

    def test_recall_and_faithfulness_metrics(self) -> None:
        self.assertEqual(0.5, recall_at_k(["a", "x", "b"], ["a", "b"], 2))
        grounded = faithfulness_score("高血压需要监测血压。[1]", ["高血压需要监测血压。"])
        hallucinated = faithfulness_score("火箭需要液氧燃料。[1]", ["高血压需要监测血压。"])
        self.assertEqual(1.0, grounded)
        self.assertEqual(0.0, hallucinated)

    def test_thinking_is_removed_from_generation_and_evidence(self) -> None:
        self.assertEqual("最终答案", strip_thinking("<think>隐私推理</think>最终答案"))
        generator = LangChainGroundedGenerator(ThinkingLLM())
        answer = generator.generate("问题", "[1] 证据")
        self.assertEqual("需要监测血压。[1]", answer)

        hit = RetrievalHit(
            SourceDocument(
                "thinking-source",
                "问题：如何管理？\n答案：<think>隐藏分析</think>遵医嘱。",
                {"source": "docs.txt"},
            )
        )
        context, citations = build_citations([hit])
        self.assertNotIn("隐藏分析", context)
        self.assertNotIn("<think>", citations[0].excerpt)


if __name__ == "__main__":
    unittest.main()
