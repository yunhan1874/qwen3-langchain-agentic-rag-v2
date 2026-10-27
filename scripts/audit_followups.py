"""Paired BM25/end-to-end and constrained-rewrite follow-ups to audit_pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

if __package__:
    from .audit_pipeline import AuditLLM, PostingBM25, require_existing_index, sha
else:
    from audit_pipeline import AuditLLM, PostingBM25, require_existing_index, sha
from rag_v2.agent import LangChainAgenticRAG
from rag_v2.cli import build_engine
from rag_v2.config import RAGConfig
from rag_v2.embeddings import HashingEmbedder
from rag_v2.models import SourceDocument
from rag_v2.query_rewrite import LLMQueryRewriter
from rag_v2.retrieval import ChromaDenseIndex
from rag_v2.retrieval_base import metadata_matches
from rag_v2.vllm_client import VLLMCompletionLLM


def stale_index_probe():
    class Collection:
        def __init__(self):
            self.ids = []
        def count(self):
            return len(self.ids)
        def add(self, ids, **kwargs):
            self.ids.extend(ids)
    collection = Collection()
    client = types.SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    fake = types.SimpleNamespace(PersistentClient=lambda **kwargs: client)
    with patch.dict(sys.modules, {'chromadb': fake}):
        ChromaDenseIndex([SourceDocument('old', '旧证据')], HashingEmbedder(), 'unused', 'test')
        ChromaDenseIndex([SourceDocument('new', '新证据')], HashingEmbedder(), 'unused', 'test')
    return {'kind': 'mock_constructor_same_count_changed_ids',
            'expected_current_ids': ['new'], 'actual_ids': collection.ids,
            'stale_index_reproduced': collection.ids != ['new']}


def paired_pipeline(engine, repeats):
    reference = engine.retriever.bm25
    optimized = PostingBM25(reference)
    rag = LangChainAgenticRAG(engine)
    rows, records = [], []
    client = VLLMCompletionLLM(engine.config.vllm_base_url, engine.config.vllm_model,
                               engine.config.answer_max_new_tokens, api_key=engine.config.vllm_api_key)
    engine.generator.llm = AuditLLM(client, 'answer', records)
    engine.rewriter.llm = AuditLLM(client.bind(pipeline_kwargs={'max_new_tokens': 96}), 'rewrite', records)
    cases = [
        ('blood_pressure', '高血压患者日常如何管理血压？', {}),
        ('glucose', '糖尿病患者日常如何监测血糖？', {}),
        ('medication', '高血压患者用药需要注意什么？', {}),
        ('missing_filter', '高血压患者日常如何管理血压？', {'source': '__not_present__'}),
    ]
    for repeat in range(repeats):
        for name, query, filters in cases:
            variants = ['baseline', 'postings_precheck'] if repeat % 2 == 0 else ['postings_precheck', 'baseline']
            for variant in variants:
                engine.retriever.bm25 = reference if variant == 'baseline' else optimized
                records.clear()
                started = time.perf_counter()
                if variant != 'baseline' and filters and not any(metadata_matches(d.metadata, filters) for d in engine.retriever.documents.values()):
                    state = engine.plan_query(engine.prepare({'query': query, 'filters': filters}))
                    state.update(status='abstained', hits=[], debug_hits=[], confidence=0.0)
                    result = engine.generate(state)['result']
                else:
                    result = rag.invoke(query, filters)
                rows.append({'case': name, 'repeat': repeat + 1, 'variant': variant,
                    'seconds': time.perf_counter() - started, 'status': result.status,
                    'selected_ids': result.retrieved_ids, 'answer_sha256': sha(result.answer),
                    'llm_calls': list(records)})
    engine.retriever.bm25 = reference
    return rows


def rewrite_comparison(engine, repeats):
    client = VLLMCompletionLLM(engine.config.vllm_base_url, engine.config.vllm_model, 96,
                               api_key=engine.config.vllm_api_key)
    schema = {
        'type': 'object', 'properties': {
            'query': {'type': 'string', 'minLength': 1, 'maxLength': 60},
            'filters': {'type': 'object', 'properties': {
                'source': {'type': 'string'}, 'source_line': {'type': 'integer'},
                'document_type': {'type': 'string'}, 'department': {'type': 'string'},
                'category': {'type': 'string'}, 'year': {'type': 'integer'},
            }, 'additionalProperties': False},
        }, 'required': ['query', 'filters'], 'additionalProperties': False,
    }
    rows = []
    cases = [
        ('blood_pressure', '高血压日常如何管理'),
        ('glucose', '血糖偏高日常需要记录哪些指标'),
        ('respiratory', '出现呼吸道症状时应记录哪些信息？'),
        ('rocket', '火箭发动机的推力是多少？'),
        ('code', 'Python如何对列表进行排序？'),
        ('noise', 'ZXQVV-9981 KLMNOP是什么？'),
    ]
    # Capture the exact current rewrite prompt without invoking Qwen.
    class Capture:
        def invoke(self, text):
            self.prompt = text
            return '{"query":"capture","filters":{}}'
    capture = Capture()
    for name, query in cases:
        LLMQueryRewriter(capture).rewrite(query)
        prompt = capture.prompt
        for repeat in range(repeats):
            variants = ['plain', 'json_schema'] if repeat % 2 == 0 else ['json_schema', 'plain']
            for variant in variants:
                payload = {'model': client.model, 'prompt': prompt, 'max_tokens': 96,
                           'temperature': 0.1, 'top_p': 0.95, 'top_k': 20, 'seed': 0}
                if variant == 'json_schema':
                    payload['structured_outputs'] = {'json': schema}
                started = time.perf_counter()
                row = {'case': name, 'repeat': repeat + 1, 'variant': variant}
                try:
                    response = client._request('/completions', payload)
                    choice = response['choices'][0]
                    text = choice['text']
                    valid = False
                    try:
                        parsed = LLMQueryRewriter._json_object(text)
                        valid = isinstance(parsed.get('query'), str) and bool(parsed['query'].strip()) and isinstance(parsed.get('filters'), dict)
                    except (ValueError, TypeError, AttributeError):
                        pass
                    row.update(json_valid=valid, finish_reason=choice['finish_reason'],
                        completion_tokens=response['usage']['completion_tokens'],
                        output_sha256=sha(text))
                    # These queries are synthetic/public; retain rewrites for
                    # checking meaning, never the private medical source text.
                    if valid:
                        row['rewritten_query'] = parsed['query']
                        row['filters'] = parsed['filters']
                except RuntimeError as exc:
                    row.update(error=str(exc), json_valid=False)
                row['seconds'] = time.perf_counter() - started
                rows.append(row)
                print('REWRITE', name, variant, row['json_valid'], round(row['seconds'], 3), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    config = RAGConfig()
    if config.llm_backend != 'vllm':
        raise RuntimeError('An existing vLLM deployment is required')
    require_existing_index(config)
    engine = build_engine(config)
    result = {'stale_index_probe': stale_index_probe(), 'paired_pipeline': paired_pipeline(engine, args.repeats)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    result['rewrite_comparison'] = rewrite_comparison(engine, args.repeats)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
