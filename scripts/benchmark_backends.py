"""Compare local Transformers and an OpenAI-compatible vLLM completion server.

Run with the project's real-model environment variables. Both backends consume
the exact same raw prompt, use greedy decoding, and share the existing RAG path.
Results contain hashes/metrics, never source medical passages or model output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from urllib.request import Request, urlopen
from unittest.mock import patch

from rag_v2.agent import LangChainAgenticRAG
from rag_v2.cli import _build_llm, build_engine
from rag_v2.config import RAGConfig


QUERIES = [
    "高血压患者日常如何管理血压？",
    "糖尿病患者日常如何监测血糖？",
    "高血压患者用药需要注意什么？",
]


class TokenTimer:
    """Measure first generated token, skipping generate()'s initial prompt callback."""

    def __init__(self):
        self.prompt_seen = False
        self.first_token_time = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


class BoundBackend:
    def __init__(self, parent, max_tokens):
        self.parent, self.max_tokens = parent, max_tokens

    def invoke(self, prompt):
        return self.parent.invoke(prompt, max_tokens=self.max_tokens)


class Backend:
    def __init__(self, config, name, api_url, served_model):
        self.config, self.name = config, name
        self.api_url, self.served_model = api_url, served_model
        self.records = []
        if name == "transformers":
            self.pipe = _build_llm(config).pipeline

    def bind(self, pipeline_kwargs):
        return BoundBackend(self, pipeline_kwargs["max_new_tokens"])

    def invoke(self, prompt, max_tokens=None, fixed_length=False):
        max_tokens = max_tokens or self.config.answer_max_new_tokens
        started = time.perf_counter()
        finish_reason = None
        if self.name == "transformers":
            import torch

            tokenizer, model = self.pipe.tokenizer, self.pipe.model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            timer = TokenTimer()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    min_new_tokens=max_tokens if fixed_length else 0, streamer=timer,
                )
            torch.cuda.synchronize()
            ended = time.perf_counter()
            n_prompt = inputs.input_ids.shape[-1]
            generated = result[0][n_prompt:]
            n_output = len(generated)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            first = timer.first_token_time
            finish_reason = "length" if n_output == max_tokens else "stop"
        else:
            body = {
                "model": self.served_model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0, "top_p": 1, "top_k": -1, "stream": True,
                "stream_options": {"include_usage": True}, "seed": 0,
            }
            if fixed_length:
                body.update(ignore_eos=True, min_tokens=max_tokens)
            request = Request(
                self.api_url.rstrip("/") + "/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
            )
            pieces, first, usage = [], None, {}
            with urlopen(request, timeout=180) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    value = line[6:].strip()
                    if value == b"[DONE]":
                        break
                    chunk = json.loads(value)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        part = choice.get("text", "")
                        if part and first is None:
                            first = time.perf_counter()
                        pieces.append(part)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
            ended = time.perf_counter()
            text = "".join(pieces)
            n_prompt, n_output = usage["prompt_tokens"], usage["completion_tokens"]
        elapsed = ended - started
        record = {
            "backend": self.name,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_tokens": n_prompt, "output_tokens": n_output,
            "max_tokens": max_tokens, "fixed_length": fixed_length,
            "seconds": round(elapsed, 4),
            "ttft_seconds": round(first - started, 4) if first is not None else None,
            "tokens_per_second": round(n_output / elapsed, 2),
            "finish_reason": finish_reason,
            "raw_citation_ids": re.findall(r"\[(\d+)\]", text),
        }
        self.records.append(record)
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--served-model", default="qwen3-medical-benchmark")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = RAGConfig()
    config.validate()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    backend = Backend(config, args.backend, args.api_url, args.served_model)
    # Use one Qwen model per backend; the RAG layer still uses its usual BGE models.
    with patch("rag_v2.cli._build_llm", return_value=backend):
        engine = build_engine(config)
    rag = LangChainAgenticRAG(engine)
    captured_prompts = []
    original_invoke = backend.invoke

    def capture(prompt, **kwargs):
        captured_prompts.append(prompt)
        return original_invoke(prompt, **kwargs)

    backend.invoke = capture
    rag.invoke(QUERIES[0])  # excluded warmup
    backend.records.clear()
    runs = []
    for round_index in range(args.rounds):
        for query in QUERIES:
            backend.records.clear()
            started = time.perf_counter()
            result = rag.invoke(query)
            elapsed = time.perf_counter() - started
            assert result.status == "answered", result.agent_steps
            assert result.citations and len(result.answer) > 20
            assert len(backend.records) == 1, "Expected direct retrieval and one generation"
            calls = list(backend.records)
            valid_ids = {str(c.citation_id) for c in result.citations}
            raw_ids = calls[0]["raw_citation_ids"]
            row = {
                "query": query, "round": round_index + 1, "e2e_seconds": round(elapsed, 4),
                "status": result.status, "citation_count": len(result.citations),
                "chunk_ids": result.retrieved_ids, "answer_chars": len(result.answer),
                "raw_inline_citations_present": bool(raw_ids),
                "raw_citation_ids_valid": set(raw_ids) <= valid_ids,
                "generation": calls[0],
            }
            runs.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    fixed = []
    for _ in range(3):
        backend.invoke(captured_prompts[0], max_tokens=128, fixed_length=True)
        row = backend.records[-1]
        assert row["output_tokens"] == 128, row
        fixed.append(row)
        print("FIXED " + json.dumps(row), flush=True)
    summary = {
        "backend": args.backend, "rounds": args.rounds, "greedy": True,
        "model_path": config.model_path,
        "e2e_median_seconds": round(statistics.median(r["e2e_seconds"] for r in runs), 4),
        "generation_median_seconds": round(statistics.median(r["generation"]["seconds"] for r in runs), 4),
        "fixed_128_median_seconds": round(statistics.median(r["seconds"] for r in fixed), 4),
        "fixed_128_median_ttft_seconds": round(statistics.median(r["ttft_seconds"] for r in fixed), 4),
        "runs": runs, "fixed_length": fixed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in {"runs", "fixed_length"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
