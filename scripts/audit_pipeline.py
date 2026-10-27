"""Audit the installed RAG pipeline without changing production configuration.

Run from an installed project environment. Raw prompts, answers and source queries
are never exported. Source-title lookup is only a retrieval regression proxy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time

from rag_v2.agent import LangChainAgenticRAG
from rag_v2.citations import enforce_valid_citations
from rag_v2.cli import build_engine
from rag_v2.config import RAGConfig
from rag_v2.evaluation import faithfulness_score
from rag_v2.models import Citation
from rag_v2.retrieval_base import metadata_matches
from rag_v2.text import tokenize
from rag_v2.vllm_client import VLLMCompletionLLM


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def median(values):
    return statistics.median(values) if values else None


def require_existing_index(config):
    """Refuse conditions where build_engine would add or recreate documents."""
    chunks_path = config.index_dir / 'chunks.jsonl'
    chroma_path = config.index_dir / 'chroma'
    if not chunks_path.is_file() or not (chroma_path / 'chroma.sqlite3').is_file():
        raise RuntimeError('An existing complete index is required; audit never builds one')
    from rag_v2.semantic_chunking import load_chunks
    import chromadb
    collection = chromadb.PersistentClient(path=str(chroma_path)).get_collection(config.collection_name)
    if collection.count() != len(load_chunks(chunks_path)) or collection.count() == 0:
        raise RuntimeError('Index count mismatch; audit refuses automatic rebuild')


class AuditLLM:
    """Production sampling plus seed=0, with usage and native citations recorded."""

    def __init__(self, client, kind, records):
        self.client, self.kind, self.records = client, kind, records

    def invoke(self, prompt):
        started = time.perf_counter()
        response = self.client._request('/completions', {
            'model': self.client.model, 'prompt': prompt,
            'max_tokens': self.client.max_new_tokens, 'temperature': 0.1,
            'top_p': 0.95, 'top_k': 20, 'seed': 0, 'stream': False,
        })
        elapsed = time.perf_counter() - started
        choice = response['choices'][0]
        text = choice['text']
        self.records.append({
            'kind': self.kind, 'seconds': elapsed, 'limit': self.client.max_new_tokens,
            'prompt_sha256': sha(prompt), 'output_sha256': sha(text),
            'usage': response.get('usage', {}), 'finish_reason': choice['finish_reason'],
            'native_citation_ids': [int(x) for x in re.findall(r'\[(\d+)\]', text)],
        })
        return text

    def bind(self, *, pipeline_kwargs):
        return AuditLLM(self.client.bind(pipeline_kwargs=pipeline_kwargs), self.kind, self.records)


class PostingBM25:
    """Experimental equivalent scorer, retaining original per-term float operations."""

    def __init__(self, reference):
        self.documents = reference.documents
        self.postings = defaultdict(list)
        for index, frequencies in enumerate(reference.term_frequencies):
            length = reference.lengths[index]
            for term, frequency in frequencies.items():
                denominator = frequency + reference.k1 * (
                    1 - reference.b + reference.b * length / (reference.average_length or 1)
                )
                score = reference.idf.get(term, 0.0) * frequency * (reference.k1 + 1) / denominator
                self.postings[term].append((index, score))

    def search(self, query, k, filters=None):
        scores = defaultdict(float)
        allowed = None
        if filters:
            allowed = {i for i, d in enumerate(self.documents) if metadata_matches(d.metadata, filters)}
            if not allowed:
                return []
        for term in tokenize(query):
            for index, score in self.postings.get(term, ()):
                if allowed is None or index in allowed:
                    scores[index] += score
        rows = [(self.documents[i].document_id, score) for i, score in scores.items() if score > 0]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows[:k]


def correctness_probes():
    synthetic = [
        ('negation', '患者应服用阿司匹林。', ['患者不应服用阿司匹林。']),
        ('dose', '每天服用药物100毫克。', ['每天服用药物10毫克。']),
        ('unsupported', '地球围绕太阳运行。', ['系统仅记录血压。']),
        ('supported', '系统记录血压。', ['系统记录血压。']),
    ]
    probes = [{'case': name, 'score': faithfulness_score(answer, contexts),
               'expected_supported': name == 'supported'} for name, answer, contexts in synthetic]
    evidence = Citation(1, 'synthetic', 'synthetic', '', '系统记录血压。')
    uncited = '地球围绕太阳运行。'
    probes.append({'case': 'citation_auto_append',
                   'input_has_citation': False,
                   'output_has_citation': '[1]' in enforce_valid_citations(uncited, [evidence]),
                   'expected_supported': False})
    return probes


def profile(engine, args, output):
    records, timings = [], []
    client = VLLMCompletionLLM(engine.config.vllm_base_url, engine.config.vllm_model,
                               engine.config.answer_max_new_tokens, api_key=engine.config.vllm_api_key)
    engine.generator.llm = AuditLLM(client, 'answer', records)
    engine.rewriter.llm = AuditLLM(
        client.bind(pipeline_kwargs={'max_new_tokens': engine.config.rewrite_max_new_tokens}),
        'rewrite', records,
    )
    for obj, method, label in [
        (engine.retriever.dense, 'search', 'dense'),
        (engine.retriever.bm25, 'search', 'bm25'),
        (engine.reranker, 'rerank', 'rerank'),
        (engine.compressor, 'compress', 'compression'),
    ]:
        original = getattr(obj, method)

        def measured(*a, _fn=original, _label=label, **kw):
            started = time.perf_counter()
            result = _fn(*a, **kw)
            timings.append({'stage': _label, 'seconds': time.perf_counter() - started})
            return result

        setattr(obj, method, measured)
    rag = LangChainAgenticRAG(engine)
    cases = [
        ('blood_pressure', '高血压患者日常如何管理血压？', {}),
        ('glucose', '糖尿病患者日常如何监测血糖？', {}),
        ('medication', '高血压患者用药需要注意什么？', {}),
        ('respiratory', '出现呼吸道症状时应记录哪些信息？', {}),
        ('out_of_domain_rocket', '火箭发动机的推力是多少？', {}),
        ('out_of_domain_code', 'Python如何对列表进行排序？', {}),
        ('out_of_domain_noise', 'ZXQVV-9981 KLMNOP是什么？', {}),
        ('missing_filter', '高血压患者日常如何管理血压？', {'source': '__not_present__'}),
    ]
    rag.invoke(cases[0][1])
    output['profiles'] = []
    for repeat in range(args.repeats):
        for name, query, filters in cases:
            records.clear()
            timings.clear()
            started = time.perf_counter()
            result = rag.invoke(query, filters)
            row = {'case': name, 'repeat': repeat + 1, 'seconds': time.perf_counter() - started,
                   'status': result.status, 'confidence': result.confidence,
                   'citation_count': len(result.citations), 'answer_sha256': sha(result.answer),
                   'steps': [s.split(':', 1)[0] for s in result.agent_steps],
                   'stages': list(timings), 'llm_calls': list(records),
                   'lexical_faithfulness_proxy': faithfulness_score(result.answer, [c.excerpt for c in result.citations])}
            output['profiles'].append(row)
            print('PROFILE', name, repeat + 1, result.status, round(row['seconds'], 4), flush=True)
    # Same captured input under two budgets; no claim that shorter is equally good.
    output['answer_budgets'] = []
    for name, query, filters in cases[:4]:
        state = engine.prepare({'query': query, 'filters': filters})
        state = engine.plan_query(state)
        state = engine.agentic_retrieve(state)
        state = engine.compress(state)
        if state['status'] == 'abstained':
            continue
        for repeat in range(args.repeats):
            for limit in ([256, 128] if repeat % 2 == 0 else [128, 256]):
                records.clear()
                engine.generator.llm = AuditLLM(replace(client, max_new_tokens=limit), 'answer', records)
                result = engine.generate(state)['result']
                output['answer_budgets'].append({'case': name, 'repeat': repeat + 1,
                    'limit': limit, 'call': dict(records[-1]), 'answer_sha256': sha(result.answer)})
    engine.generator.llm = AuditLLM(client, 'answer', records)


def retrieval_audit(engine, args, output):
    # Recover known question titles from index metadata. This is self-lookup,
    # not an independent medical gold set or a generalization evaluation.
    by_parent = {}
    for document in engine.retriever.documents.values():
        parent = document.metadata.get('parent_document_id', document.document_id)
        title = str(document.metadata.get('title', ''))
        question = re.sub(r'^问题\s*[:：]\s*', '', title).strip()
        if 8 <= len(question) <= 90 and '答案' not in question:
            by_parent.setdefault(parent, question)
    population = sorted(by_parent.items())
    selected = random.Random(20261027).sample(population, min(args.lookup_cases, len(population)))
    output['lookup_case_count'] = len(selected)
    output['lookup_pool_count'] = len(population)
    output['retrieval_sweep'], output['bm25_comparison'] = [], []
    posting = PostingBM25(engine.retriever.bm25)
    pools = []
    engine.retriever.fused_k = 24
    for parent, query in selected:
        candidates = engine.retriever.retrieve(query)
        pools.append((parent, query, candidates))
        for count in (6, 12, 24):
            rows, durations = [], []
            for _ in range(args.repeats):
                started = time.perf_counter()
                rows = engine.reranker.rerank(query, candidates[:count], 5)
                durations.append(time.perf_counter() - started)
            accepted = [h for h in rows if h.rerank_score >= engine.config.min_relevance_score][:3]
            output['retrieval_sweep'].append({'query_sha256': sha(query), 'parent_id': parent,
                'candidate_limit': count, 'candidate_count': len(candidates[:count]),
                'seconds': durations, 'source_in_candidates': any(h.document.metadata.get('parent_document_id') == parent for h in candidates[:count]),
                'source_in_selected': any(h.document.metadata.get('parent_document_id') == parent for h in accepted),
                'selected_ids': [h.document.document_id for h in accepted],
                'selected_scores': [h.rerank_score for h in accepted]})
        for filters in ({}, {'document_type': 'medical_qa'}, {'source': '__not_present__'}):
            timings = {'original': [], 'postings': []}
            originals = optimized = []
            for _ in range(5):
                for name, scorer in [('original', engine.retriever.bm25), ('postings', posting)]:
                    started = time.perf_counter()
                    values = scorer.search(query, 12, filters)
                    timings[name].append(time.perf_counter() - started)
                    if name == 'original':
                        originals = values
                    else:
                        optimized = values
            output['bm25_comparison'].append({'query_sha256': sha(query), 'filter_kind': str(filters),
                'exact_results_equal': originals == optimized, 'seconds': timings})
    engine.retriever.fused_k = 12
    output['precision'] = []
    import torch
    dtype = str(next(engine.reranker.model.model.parameters()).dtype)
    baselines = {}
    for parent, query, candidates in pools:
        rows = engine.reranker.rerank(query, candidates[:12], 5)
        baselines[sha(query)] = rows
    engine.reranker.model.model.half()  # Only this audit's model instance.
    for _ in range(2):
        if pools:
            engine.reranker.rerank(pools[0][1], pools[0][2][:12], 5)
    for parent, query, candidates in pools:
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            rows = engine.reranker.rerank(query, candidates[:12], 5)
            durations.append(time.perf_counter() - started)
        base = baselines[sha(query)]
        base_ids = [h.document.document_id for h in base if h.rerank_score >= engine.config.min_relevance_score][:3]
        half_ids = [h.document.document_id for h in rows if h.rerank_score >= engine.config.min_relevance_score][:3]
        scores = {h.document.document_id: h.rerank_score for h in base}
        output['precision'].append({'query_sha256': sha(query), 'original_dtype': dtype,
            'test_dtype': str(next(engine.reranker.model.model.parameters()).dtype),
            'seconds': durations, 'selected_order_equal': base_ids == half_ids,
            'selected_set_equal': set(base_ids) == set(half_ids),
            'top5_common_score_max_delta': max([abs(h.rerank_score - scores[h.document.document_id]) for h in rows if h.document.document_id in scores] or [0])})
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--lookup-cases', type=int, default=24)
    args = parser.parse_args()
    if args.repeats < 1 or args.lookup_cases < 1:
        parser.error('positive repeats and lookup-cases required')
    config = RAGConfig()
    if config.llm_backend != 'vllm':
        raise RuntimeError('Set RAG_LLM_BACKEND=vllm; audit never loads extra Qwen weights')
    require_existing_index(config)
    output = {'correctness_probes': correctness_probes(), 'repeats': args.repeats,
              'notes': ['self-lookup is a regression proxy, not independent Recall@K',
                        'sampling matches production plus fixed seed=0',
                        'raw medical queries, prompts and outputs are excluded']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(config)
    output['chunk_count'] = len(engine.retriever.documents)
    # Persist useful diagnostics even if a later experiment fails.
    profile(engine, args, output)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('START retrieval sweep', flush=True)
    retrieval_audit(engine, args, output)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', args.output.name, flush=True)


if __name__ == '__main__':
    main()
