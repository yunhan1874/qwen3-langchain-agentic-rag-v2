from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_v2.agent import AgenticRAGEngine
from rag_v2.citations import build_citations, enforce_valid_citations
from rag_v2.compression import ExtractiveContextCompressor
from rag_v2.config import RAGConfig
from rag_v2.embeddings import HashingEmbedder
from rag_v2.evaluation import EvaluationCase, RAGEvaluator, exact_sentence_support_score, lexical_support_score
from rag_v2.index_cache import atomic_json, index_identity, manifest_matches, version_directory
from rag_v2.models import Citation, RAGAnswer, RetrievalHit, SourceDocument
from rag_v2.retrieval import BM25Index, HybridRetriever, InMemoryDenseIndex
from rag_v2.text import strip_thinking, tokenize


def slow_bm25(index, query, filters):
    from rag_v2.retrieval import metadata_matches
    rows = []
    for i, doc in enumerate(index.documents):
        if not metadata_matches(doc.metadata, filters):
            continue
        score = 0.0
        for term in tokenize(query):
            frequency = index.term_frequencies[i].get(term, 0)
            if frequency:
                denominator = frequency + index.k1 * (1 - index.b + index.b * index.lengths[i] / (index.average_length or 1))
                score += index.idf[term] * frequency * (index.k1 + 1) / denominator
        if score > 0:
            rows.append((doc.document_id, score))
    return sorted(rows, key=lambda x: (-x[1], x[0]))[:12]


class OptimizationRegressionTests(unittest.TestCase):
    def test_postings_preserve_exact_scores_duplicates_ties_and_filters(self):
        rng = random.Random(81)
        docs = [SourceDocument(str(i), ''.join(rng.choices('甲乙丙丁12AB ', k=rng.randint(0, 90))),
                               {'source': str(i % 4)}) for i in range(50)]
        index = BM25Index(docs)
        for _ in range(60):
            query = ''.join(rng.choices('甲乙丙丁12AB ', k=rng.randint(0, 40)))
            for filters in ({}, {'source': '2'}, {'source': ['1', '3']}, {'source': 'absent'}):
                self.assertEqual(slow_bm25(index, query, filters), index.search(query, 12, filters))

    def engine(self):
        docs = [SourceDocument('id', '系统记录血压。', {'source': 's'})]
        dense = InMemoryDenseIndex(docs, HashingEmbedder())
        retriever = HybridRetriever(docs, dense)
        reranker = Mock()
        reranker.rerank.side_effect = lambda q, hits, k: [replace(h, rerank_score=0.9) for h in hits[:k]]
        return AgenticRAGEngine(retriever, reranker, Mock(), ExtractiveContextCompressor(), Mock(), RAGConfig())

    def test_impossible_explicit_filter_skips_both_routes_and_llm(self):
        engine = self.engine()
        with patch.object(engine.retriever, 'retrieve') as retrieve:
            answer = engine.run('系统如何记录', {'source': 'absent'})
        retrieve.assert_not_called()
        engine.rewriter.rewrite.assert_not_called()
        engine.generator.generate.assert_not_called()
        self.assertEqual('abstained', answer.status)
        self.assertIn('agent_stop:no_matching_explicit_filters', answer.agent_steps)

    def test_missing_citations_remain_missing_and_model_citations_are_unverified(self):
        engine = self.engine()
        engine.generator.generate.return_value = '系统记录血压。'
        answer = engine.run('系统记录血压')
        self.assertEqual('missing', answer.citation_status)
        self.assertNotIn('[1]', answer.answer)
        engine.generator.generate.return_value = '系统记录血压。[1] [99]'
        answer = engine.run('系统记录血压')
        self.assertEqual('present_unverified', answer.citation_status)
        self.assertNotIn('[99]', answer.answer)

    def test_empty_generation_is_not_a_successful_answer(self):
        engine = self.engine()
        engine.generator.generate.return_value = ''
        self.assertEqual('generation_failed', engine.run('系统记录血压').status)

    def test_thinking_removes_complete_unclosed_multiple_and_prefilled_blocks(self):
        for raw, expected in [('<think>hidden', ''), ('prefix<think>hidden', 'prefix'),
                              ('<THINK>hidden</THINK>visible', 'visible'),
                              ('<think>a</think>visible<think>b', 'visible'),
                              ('hidden</think>visible', 'visible')]:
            self.assertEqual(expected, strip_thinking(raw))

    def test_evidence_preserves_text_after_excerpt_limit(self):
        text = '系统记录。' * 40 + '最终证据完整保留。'
        _, citations = build_citations([RetrievalHit(SourceDocument('id', text))])
        self.assertNotIn('最终证据', citations[0].excerpt)
        self.assertEqual(text, citations[0].evidence)
        self.assertEqual('无依据。', enforce_valid_citations('无依据。[99]', citations))

    def test_proxy_metrics_do_not_claim_semantic_faithfulness(self):
        for answer, evidence in [('患者应服用药物。', '患者不应服用药物。'),
                                 ('每天服用药物100毫克。', '每天服用药物10毫克。')]:
            self.assertEqual(1.0, lexical_support_score(answer, [evidence]))
            self.assertEqual(0.0, exact_sentence_support_score(answer, [evidence]))
        self.assertEqual(0.0, lexical_support_score('', ['evidence']))
        self.assertEqual(0.0, exact_sentence_support_score('', ['evidence']))

    def test_evaluation_uses_lcel_full_evidence_and_excludes_abstentions_from_support(self):
        engine = Mock()
        citation = Citation(1, 'id', 'source', '', 'unrelated excerpt', '系统记录血压。')
        def result(status, answer, citations):
            return RAGAnswer(answer, citations, 'q', 'q', status, [], ['id'], 0.9)
        responses = [result('answered', '系统记录血压。[1]', [citation]),
                     result('abstained', '证据不足。', []),
                     result('generation_failed', '生成失败。', [])]
        with patch('rag_v2.evaluation.LangChainAgenticRAG') as facade:
            facade.return_value.invoke.side_effect = responses
            report = RAGEvaluator(engine).evaluate([EvaluationCase('q', ['id'], {})] * 3).as_dict()
        engine.run.assert_not_called()
        self.assertIsNone(report['faithfulness'])
        self.assertEqual(1, report['answered_count'])
        self.assertEqual(1.0, report['exact_sentence_support'])
        self.assertEqual(1.0, report['citation_rate'])
        self.assertEqual(1 / 3, report['abstention_rate'])

    def test_index_versions_follow_source_model_and_chunk_changes_and_keep_old_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / 'docs.txt'
            docs.write_text('old', encoding='utf-8')
            model = root / 'model'
            model.mkdir()
            weights = model / 'model.safetensors'
            weights.write_bytes(b'old')
            config = replace(RAGConfig(), docs_path=docs, index_dir=root / 'index', embedding_model=str(model))
            identity = index_identity(config)
            old = version_directory(config, identity)
            atomic_json(old / 'manifest.json', identity)
            self.assertTrue(manifest_matches(old, identity))
            docs.write_text('new', encoding='utf-8')
            source_changed = version_directory(config)
            self.assertNotEqual(old, source_changed)
            weights.write_bytes(b'new')
            self.assertNotEqual(source_changed, version_directory(config))
            self.assertNotEqual(version_directory(config), version_directory(replace(config, min_chunk_chars=130)))
            self.assertTrue((old / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
