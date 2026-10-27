"""Compare this release with a private source snapshot; never exports model text."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

if __package__:
    from .audit_pipeline import AuditLLM
else:
    from audit_pipeline import AuditLLM
from rag_v2.agent import LangChainAgenticRAG
from rag_v2.cli import build_engine
from rag_v2.config import RAGConfig
from rag_v2.embeddings import HashingEmbedder
from rag_v2.evaluation import exact_sentence_support_score, lexical_support_score
from rag_v2.index_cache import manifest_matches, index_identity, version_directory
from rag_v2.models import SourceDocument
from rag_v2.retrieval import ChromaDenseIndex
from rag_v2.text import strip_thinking
from rag_v2.vllm_client import VLLMCompletionLLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('baseline_rag', args.baseline_package / '__init__.py',
            submodule_search_locations=[str(args.baseline_package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['baseline_rag'] = module
    spec.loader.exec_module(module)
    old_cli = importlib.import_module('baseline_rag.cli')
    old_config = importlib.import_module('baseline_rag.config')
    old_agent = importlib.import_module('baseline_rag.agent')
    config = RAGConfig()
    if config.llm_backend != 'vllm':
        raise RuntimeError('vLLM deployment required')
    old = old_cli.build_engine(old_config.RAGConfig())
    new = build_engine(config)
    old_ids = set(old.retriever.documents)
    new_ids = set(new.retriever.documents)
    assert old_ids == new_ids, 'Chunk identity changed; investigate before deployment'
    records = []
    client = VLLMCompletionLLM(config.vllm_base_url, config.vllm_model, config.answer_max_new_tokens,
                               api_key=config.vllm_api_key)
    for engine in (old, new):
        engine.generator.llm = AuditLLM(client, 'answer', records)
        engine.rewriter.llm = AuditLLM(replace(client, max_new_tokens=config.rewrite_max_new_tokens), 'rewrite', records)
    facades = {'baseline': old_agent.LangChainAgenticRAG(old), 'optimized': LangChainAgenticRAG(new)}
    cases = [('blood_pressure', '高血压患者日常如何管理血压？', {}),
             ('glucose', '糖尿病患者日常如何监测血糖？', {}),
             ('medication', '高血压患者用药需要注意什么？', {}),
             ('rocket', '火箭发动机的推力是多少？', {}),
             ('missing_filter', '高血压患者日常如何管理血压？', {'source': '__not_present__'})]
    for facade in facades.values():
        facade.invoke(cases[0][1])
    rows = []
    for repeat in range(3):
        for name, query, filters in cases:
            for variant in (list(facades) if repeat % 2 == 0 else list(reversed(facades))):
                records.clear()
                started = time.perf_counter()
                answer = facades[variant].invoke(query, filters)
                row = {'case': name, 'repeat': repeat + 1, 'variant': variant,
                       'seconds': time.perf_counter() - started, 'status': answer.status,
                       'selected_ids': answer.retrieved_ids, 'llm_calls': list(records),
                       'citation_status': getattr(answer, 'citation_status', 'legacy_auto_append')}
                rows.append(row)
                print('VERIFY', name, variant, round(row['seconds'], 4), answer.status, flush=True)
    for name, _, _ in cases:
        for repeat in range(1, 4):
            pair = [r for r in rows if r['case'] == name and r['repeat'] == repeat]
            assert pair[0]['selected_ids'] == pair[1]['selected_ids']
            assert pair[0]['status'] == pair[1]['status']
            outputs = [[c['output_sha256'] for c in r['llm_calls'] if c['kind'] == 'answer'] for r in pair]
            assert outputs[0] == outputs[1], 'Raw model answer changed'
    bm25 = []
    for name, query, filters in cases:
        durations = {'baseline': [], 'optimized': []}
        expected = old.retriever.bm25.search(query, 12, filters)
        assert expected == new.retriever.bm25.search(query, 12, filters)
        for _ in range(10):
            for label, index in [('baseline', old.retriever.bm25), ('optimized', new.retriever.bm25)]:
                begin = time.perf_counter()
                assert index.search(query, 12, filters) == expected
                durations[label].append(time.perf_counter() - begin)
        bm25.append({'case': name, 'seconds': durations, 'exact_results_equal': True})
    with tempfile.TemporaryDirectory() as folder:
        dense = ChromaDenseIndex([SourceDocument('old', 'synthetic old')], HashingEmbedder(), folder, 'release-test')
        try:
            ChromaDenseIndex([SourceDocument('new', 'synthetic new')], HashingEmbedder(), folder, 'release-test')
        except RuntimeError:
            stale_rejected = True
        else:
            stale_rejected = False
        old_preserved = dense.collection.get()['ids'] == ['old']
    assert stale_rejected and old_preserved
    result = {'date': '2026-10-27', 'chunk_count': len(new_ids), 'chunk_ids_equal': True,
              'profiles': rows, 'bm25': bm25,
              'real_chroma_stale_index_rejected': stale_rejected,
              'old_synthetic_index_preserved': old_preserved,
              'new_manifest_matches': manifest_matches(version_directory(config), index_identity(config)),
              'old_index_retained': (config.index_dir / 'chroma' / 'chroma.sqlite3').exists(),
              'unclosed_thinking_removed': strip_thinking('<think>unfinished') == '',
              'contradictions': [{'case': name, 'lexical_proxy': lexical_support_score(a, [c]),
                                  'exact_sentence_support': exact_sentence_support_score(a, [c])}
                   for name, a, c in [('negation', '患者应服用药物。', '患者不应服用药物。'),
                                      ('dose', '每天服用药物100毫克。', '每天服用药物10毫克。')]],
              'notes': ['Raw model answers and private prompts are excluded.',
                        'Citation postprocessing intentionally changes; raw answer hashes match.',
                        'No medical correctness or semantic entailment evaluation.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, _, _ in cases:
        print('MEDIAN', name, {v: statistics.median(r['seconds'] for r in rows if r['case'] == name and r['variant'] == v) for v in facades}, flush=True)
    print('ALL_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
