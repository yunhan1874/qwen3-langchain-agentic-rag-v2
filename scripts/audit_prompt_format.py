"""Compare raw completion and Qwen chat templates on the same RAG evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from transformers import AutoTokenizer

if __package__:
    from .audit_pipeline import require_existing_index
else:
    from audit_pipeline import require_existing_index

from rag_v2.citations import build_citations
from rag_v2.cli import build_engine
from rag_v2.config import RAGConfig
from rag_v2.generation import LangChainGroundedGenerator
from rag_v2.query_rewrite import LLMQueryRewriter
from rag_v2.text import strip_thinking
from rag_v2.vllm_client import VLLMCompletionLLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = RAGConfig()
    if config.llm_backend != 'vllm':
        raise RuntimeError('An existing vLLM deployment is required')
    require_existing_index(config)
    engine = build_engine(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False, local_files_only=True)
    client = VLLMCompletionLLM(config.vllm_base_url, config.vllm_model, api_key=config.vllm_api_key)
    class Capture:
        def invoke(self, text):
            self.prompt = text
            return '{}'
    capture = Capture()
    rows = []
    cases = [('blood_pressure', '高血压患者日常如何管理血压？'),
             ('glucose', '糖尿病患者日常如何监测血糖？'),
             ('medication', '高血压患者用药需要注意什么？')]
    for kind in ('answer', 'rewrite'):
        for name, query in cases:
            valid_ids = set()
            if kind == 'answer':
                state = engine.compress(engine.agentic_retrieve(engine.plan_query(engine.prepare(query))))
                context, citations = build_citations(state['compressed_hits'])
                valid_ids = {c.citation_id for c in citations}
                LangChainGroundedGenerator(capture).generate(query, context)
            else:
                LLMQueryRewriter(capture).rewrite(query)
            raw = capture.prompt
            prompts = {
                'raw': raw,
                'chat_default': tokenizer.apply_chat_template([{'role': 'user', 'content': raw}], tokenize=False, add_generation_prompt=True),
                'chat_no_thinking': tokenizer.apply_chat_template([{'role': 'user', 'content': raw}], tokenize=False, add_generation_prompt=True, enable_thinking=False),
            }
            for repeat in range(2):
                for variant in (list(prompts) if repeat == 0 else list(reversed(prompts))):
                    started = time.perf_counter()
                    response = client._request('/completions', {'model': client.model, 'prompt': prompts[variant],
                        'max_tokens': 256 if kind == 'answer' else 96, 'temperature': 0.1,
                        'top_p': 0.95, 'top_k': 20, 'seed': 0})
                    elapsed = time.perf_counter() - started
                    choice = response['choices'][0]
                    raw_output = choice['text']
                    visible = strip_thinking(raw_output)
                    citation_ids = {int(n) for n in re.findall(r'\[(\d+)\]', visible)}
                    json_valid = False
                    if kind == 'rewrite':
                        try:
                            parsed = LLMQueryRewriter._json_object(visible)
                            json_valid = isinstance(parsed.get('query'), str) and bool(parsed['query'].strip()) and isinstance(parsed.get('filters'), dict)
                        except (ValueError, TypeError, AttributeError):
                            pass
                    rows.append({'kind': kind, 'case': name, 'repeat': repeat + 1, 'variant': variant,
                        'seconds': elapsed, 'usage': response['usage'], 'finish_reason': choice['finish_reason'],
                        'has_native_valid_citation': bool(citation_ids & valid_ids), 'json_valid': json_valid,
                        'unclosed_think': '<think>' in raw_output and '</think>' not in raw_output,
                        'empty_visible_answer': not visible.strip()})
                    print('FORMAT', kind, name, variant, round(elapsed, 3), choice['finish_reason'], bool(citation_ids & valid_ids), json_valid, flush=True)
    result = {'rows': rows,
              'unclosed_think_synthetic_leaks': strip_thinking('<think>synthetic unfinished reasoning') == 'synthetic unfinished reasoning',
              'notes': ['Formatting probe only, not a medical correctness evaluation.',
                        'No system role was added; original raw prompt remains intact in user content.',
                        'chat_no_thinking also adds Qwen no-thinking prefix, not a pure format-only ablation.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
