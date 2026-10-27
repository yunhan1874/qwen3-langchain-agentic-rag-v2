from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag_v2.agent import AgenticRAGEngine
from rag_v2.cli import _build_llm, build_engine
from rag_v2.compression import ExtractiveContextCompressor
from rag_v2.config import RAGConfig
from rag_v2.models import RetrievalHit, SourceDocument
from rag_v2.query_rewrite import LLMQueryRewriter, QueryPlan


class LatencyPathTests(unittest.TestCase):
    def make_engine(self, responses, max_steps=3):
        hit = RetrievalHit(
            SourceDocument("evidence", "按要求记录并监测。", {"source": "test"}),
            rerank_score=0.9,
        )
        retriever = Mock()
        retriever.retrieve.side_effect = responses
        rewriter = Mock()
        rewriter.rewrite.return_value = QueryPlan("问题", "改写查询", {})
        rewriter.retry_query.return_value = "第二次改写"
        generator = Mock()
        generator.generate.return_value = "按要求记录并监测。[1]"
        reranker = Mock()
        reranker.rerank.side_effect = lambda query, hits, k: hits[:k]
        engine = AgenticRAGEngine(
            retriever, reranker, rewriter, ExtractiveContextCompressor(100),
            generator, replace(RAGConfig(), max_agent_steps=max_steps),
        )
        return engine, hit

    def test_sufficient_first_retrieval_skips_all_rewriting(self):
        engine, hit = self.make_engine([])
        engine.retriever.retrieve.side_effect = [[hit]]
        result = engine.run("明确问题", {"source": "test"})
        engine.retriever.retrieve.assert_called_once_with("明确问题", {"source": "test"})
        engine.rewriter.rewrite.assert_not_called()
        engine.rewriter.retry_query.assert_not_called()
        self.assertEqual("answered", result.status)
        self.assertTrue(result.citations)
        self.assertEqual("明确问题", result.rewritten_query)

    def test_low_recall_rewrites_then_retrieves_and_preserves_explicit_filters(self):
        engine, hit = self.make_engine([])
        engine.retriever.retrieve.side_effect = [[], [hit]]
        engine.rewriter.rewrite.return_value = QueryPlan(
            "问题", "改写查询", {"source": "wrong", "year": 2026}
        )
        result = engine.run("问题", {"source": "test"})
        engine.rewriter.rewrite.assert_called_once_with("问题", {"source": "test"})
        self.assertEqual(
            [("问题", {"source": "test"}),
             ("改写查询", {"source": "test", "year": 2026})],
            [call.args for call in engine.retriever.retrieve.call_args_list],
        )
        engine.rewriter.retry_query.assert_not_called()
        self.assertEqual("改写查询", result.rewritten_query)
        self.assertEqual("answered", result.status)

    def test_exhausted_retrieval_abstains_without_a_final_unused_llm_call(self):
        engine, _ = self.make_engine([[], [], []])
        result = engine.run("问题")
        self.assertEqual(3, engine.retriever.retrieve.call_count)
        engine.rewriter.rewrite.assert_called_once()
        engine.rewriter.retry_query.assert_called_once()
        engine.generator.generate.assert_not_called()
        self.assertEqual("abstained", result.status)
        self.assertEqual([], result.citations)
        self.assertIn("agent_stop:max_attempts", result.agent_steps)

    def test_nonempty_but_low_scoring_hits_still_trigger_rewrite(self):
        engine, hit = self.make_engine([])
        weak_hit = replace(hit, rerank_score=0.1)
        engine.retriever.retrieve.side_effect = [[weak_hit], [hit]]
        result = engine.run("问题")
        engine.rewriter.rewrite.assert_called_once()
        self.assertEqual(2, engine.retriever.retrieve.call_count)
        self.assertEqual("answered", result.status)
        self.assertEqual("改写查询", result.rewritten_query)

    def test_single_attempt_does_not_rewrite_after_failure(self):
        engine, _ = self.make_engine([[]], max_steps=1)
        result = engine.run("问题")
        engine.rewriter.rewrite.assert_not_called()
        engine.rewriter.retry_query.assert_not_called()
        self.assertEqual("abstained", result.status)

    def test_duplicate_rewrite_stops_without_retrieving_same_query_twice(self):
        engine, _ = self.make_engine([[]])
        engine.rewriter.rewrite.return_value = QueryPlan("问题", "问题", {})
        result = engine.run("问题")
        engine.retriever.retrieve.assert_called_once()
        engine.rewriter.retry_query.assert_not_called()
        self.assertIn("agent_stop:duplicate_query", result.agent_steps)

    def test_truncated_or_invalid_rewrite_falls_back_without_extra_generation(self):
        for text in ['{"query":"未完成', '{"query":123}', '{"query":""}']:
            with self.subTest(text=text):
                llm = Mock()
                llm.invoke.return_value = text
                result = LLMQueryRewriter(llm).rewrite("请问明确问题", {"source": "test"})
                self.assertEqual("明确问题", result.rewritten_query)
                self.assertEqual({"source": "test"}, result.filters)
                llm.invoke.assert_called_once()

    def test_pipeline_caps_answer_tokens_and_shares_model_for_rewrite(self):
        config = replace(RAGConfig(), answer_max_new_tokens=180, rewrite_max_new_tokens=64)
        pipeline_factory = Mock(return_value=object())
        hf_llm = Mock()
        hf_wrapper = Mock(return_value=hf_llm)
        modules = {
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=Mock(), AutoTokenizer=Mock(), pipeline=pipeline_factory
            ),
            "langchain_huggingface": SimpleNamespace(HuggingFacePipeline=hf_wrapper),
        }
        with patch.dict("sys.modules", modules):
            self.assertIs(hf_llm, _build_llm(config))
        self.assertEqual(180, pipeline_factory.call_args.kwargs["max_new_tokens"])
        self.assertNotIn("max_length", pipeline_factory.call_args.kwargs)

        with (
            patch("rag_v2.cli._build_llm", return_value=hf_llm) as loader,
            patch("rag_v2.cli.index_identity", return_value={}),
            patch("rag_v2.cli.atomic_json"),
            patch("rag_v2.cli.os.replace"),
            patch("rag_v2.cli.SentenceTransformerEmbedder"),
            patch("rag_v2.cli.load_source_documents", return_value=[]),
            patch("rag_v2.cli.SemanticChunker"),
            patch("rag_v2.cli.save_chunks"),
            patch("rag_v2.cli.ChromaDenseIndex"),
            patch("rag_v2.cli.HybridRetriever"),
            patch("rag_v2.cli.CrossEncoderReranker"),
        ):
            engine = build_engine(config, rebuild_index=True)
        loader.assert_called_once_with(config)
        hf_llm.bind.assert_called_once_with(pipeline_kwargs={"max_new_tokens": 64})
        self.assertIs(engine.generator.llm, hf_llm)
        self.assertIs(engine.rewriter.llm, hf_llm.bind.return_value)

    def test_generation_budgets_must_be_positive(self):
        for name in ("rewrite_max_new_tokens", "answer_max_new_tokens"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(RAGConfig(), **{name: 0}).validate()


if __name__ == "__main__":
    unittest.main()
