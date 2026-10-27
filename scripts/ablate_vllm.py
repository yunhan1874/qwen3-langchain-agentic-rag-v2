"""Controlled vLLM 0.11 ablations on a private localhost test server.

The production service is never modified. Medical prompt fixtures remain in
the ignored output directory; exported results contain hashes and timings only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import time
from urllib.request import Request, urlopen


VARIANTS = [
    {"name": "baseline", "level": 3, "graph": "FULL_AND_PIECEWISE", "attention": "FLASH_ATTN", "prefix_cache": False},
    {"name": "no_graph", "level": 3, "graph": "NONE", "attention": "FLASH_ATTN", "prefix_cache": False},
    {"name": "no_compile_no_graph", "level": 0, "graph": "NONE", "attention": "FLASH_ATTN", "prefix_cache": False},
    {"name": "no_compile_decode_graph", "level": 0, "graph": "FULL_DECODE_ONLY", "attention": "FLASH_ATTN", "prefix_cache": False},
    {"name": "triton_attention", "level": 3, "graph": "FULL_AND_PIECEWISE", "attention": "TRITON_ATTN", "prefix_cache": False},
    {"name": "prefix_cache_on", "level": 3, "graph": "FULL_AND_PIECEWISE", "attention": "FLASH_ATTN", "prefix_cache": True},
    {"name": "baseline_repeat", "level": 3, "graph": "FULL_AND_PIECEWISE", "attention": "FLASH_ATTN", "prefix_cache": False},
]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def request_json(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        text = response.read()
        return json.loads(text) if text else {}


def prepare_fixtures(args):
    from dataclasses import replace
    from unittest.mock import patch
    from transformers import AutoTokenizer
    from rag_v2.agent import LangChainAgenticRAG
    from rag_v2.cli import build_engine
    from rag_v2.config import RAGConfig

    class Capture:
        prompt = None

        def invoke(self, prompt):
            self.prompt = prompt
            return "记录血压并遵循医生建议。[1]"

        def bind(self, **kwargs):
            return self

    capture = Capture()
    config = replace(RAGConfig(), model_path=args.model)
    with patch("rag_v2.cli._build_llm", return_value=capture):
        engine = build_engine(config)
    result = LangChainAgenticRAG(engine).invoke("高血压患者日常如何管理血压？")
    assert result.status == "answered" and capture.prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    short = capture.prompt
    prefix = tokenizer.decode(tokenizer.encode(short * 6)[:2400], skip_special_tokens=False)
    cases = []
    for name, prompt in [("short", short), ("long", prefix + "\n\n" + short)]:
        n = len(tokenizer.encode(prompt))
        assert n + 128 <= 4096
        cases.append({"name": name, "prompt": prompt, "prompt_tokens": n, "prompt_sha256": sha(prompt)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "private_prompts.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print("FIXTURES " + json.dumps([{k: v for k, v in c.items() if k != "prompt"} for c in cases]), flush=True)
    del engine
    gc.collect()


def completion(base, case):
    body = {
        "model": "qwen3-ablation", "prompt": case["prompt"], "max_tokens": 128,
        "min_tokens": 128, "ignore_eos": True, "temperature": 0,
        "top_k": -1, "top_p": 1, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first, pieces, usage = None, [], {}
    request = Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("text", "")
                if text and first is None:
                    first = time.perf_counter()
                pieces.append(text)
    ended = time.perf_counter()
    assert usage["completion_tokens"] == 128, usage
    assert usage["prompt_tokens"] == case["prompt_tokens"], usage
    assert first is not None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "case": case["name"], "prompt_tokens": usage["prompt_tokens"],
        "prompt_sha256": case["prompt_sha256"], "output_tokens": 128,
        "output_sha256": sha("".join(pieces)), "cached_tokens": cached,
        "seconds": round(ended - started, 6), "ttft_seconds": round(first - started, 6),
        "decode_seconds": round(ended - first, 6),
        "tokens_per_second": round(128 / (ended - started), 3),
    }


def port_free(port):
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) != 0


def stop_server(process, port):
    if process.poll() is None:
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    for _ in range(30):
        if port_free(port):
            return
        time.sleep(0.2)
    raise RuntimeError("Ablation test port did not close")


def evidence(log_path):
    text = log_path.read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return [line for line in text.splitlines() if any(word in line for word in (
        "compilation_config=", "Using Flash Attention backend", "Using Triton backend",
        "Graph capturing finished", "Capturing CUDA graphs", "torch.compile takes",
        "setting cudagraph_mode", "compilation level", "Available KV cache memory",
    ))][-20:]


def run_variant(args, variant, cases):
    name = variant["name"]
    assert port_free(args.port), "Test port is in use; production services will not be touched"
    base = f"http://127.0.0.1:{args.port}"
    compilation = {
        "level": variant["level"], "cudagraph_mode": variant["graph"],
        "cudagraph_capture_sizes": [1, 2, 4],
    }
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", "qwen3-ablation",
        "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16",
        "--max-model-len", "4096", "--max-num-seqs", "4",
        "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", str(args.memory_fraction),
        "--enable-prefix-caching" if variant["prefix_cache"] else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details", "--disable-log-requests", "--seed", "0",
        "--compilation-config", json.dumps(compilation),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PYTHONNOUSERSITE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_USE_V1="1",
               VLLM_ATTENTION_BACKEND=variant["attention"])
    # v0.11 registers /reset_prefix_cache only in development mode. This flag
    # applies solely to the private, short-lived test server on localhost.
    env["VLLM_SERVER_DEV_MODE"] = "1" if variant["prefix_cache"] else "0"
    log_path = args.output_dir / f"{name}.log"
    record = {"variant": variant, "command": command, "samples": [],
              "test_dev_endpoints": variant["prefix_cache"]}
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, start_new_session=True, close_fds=True)
    print("START " + name + " pid=" + str(process.pid), flush=True)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Test engine exited during startup; inspect its log")
            try:
                with urlopen(base + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Test engine startup timed out")
        record["startup_seconds"] = round(time.perf_counter() - started, 3)
        # Warm both shapes before recording. Prefix cache is cleared for cold measurements.
        for case in cases:
            completion(base, case)
            completion(base, case)
        for index in range(args.repeats):
            ordered = cases if index % 2 == 0 else list(reversed(cases))
            for case in ordered:
                modes = ("cold", "warm") if variant["prefix_cache"] else ("cache_off",)
                for cache_state in modes:
                    if cache_state == "cold":
                        request_json(base, "/reset_prefix_cache", {})
                    sample = completion(base, case)
                    sample.update(repeat=index + 1, cache_state=cache_state)
                    if cache_state == "cold":
                        assert sample["cached_tokens"] == 0, sample
                    if cache_state == "warm":
                        assert sample["cached_tokens"] > 0, sample
                    record["samples"].append(sample)
        if name == "baseline":
            # Continuous batching probe: same server and prompt, 4 independent clients.
            from concurrent.futures import ThreadPoolExecutor

            case = cases[0]
            sequential = []
            for _ in range(3):
                begin = time.perf_counter()
                rows = [completion(base, case) for _ in range(4)]
                sequential.append({"seconds": time.perf_counter() - begin, "requests": rows})
            parallel = []
            for _ in range(3):
                begin = time.perf_counter()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    rows = list(pool.map(lambda _: completion(base, case), range(4)))
                parallel.append({"seconds": time.perf_counter() - begin, "requests": rows})
            record["batching_probe"] = {"sequential": sequential, "concurrent_4": parallel}
        record["status"] = "success"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_server(process, args.port)
        record["log_evidence"] = evidence(log_path)
        (args.output_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"name": name, "status": record["status"]}
    if record["status"] == "success":
        summary["medians"] = {}
        for case in cases:
            for mode in sorted({x["cache_state"] for x in record["samples"]}):
                rows = [x for x in record["samples"] if x["case"] == case["name"] and x["cache_state"] == mode]
                summary["medians"][case["name"] + "/" + mode] = {
                    "seconds": round(statistics.median(x["seconds"] for x in rows), 6),
                    "ttft_seconds": round(statistics.median(x["ttft_seconds"] for x in rows), 6),
                    "cached_tokens": sorted({x["cached_tokens"] for x in rows}),
                    "output_hashes": sorted({x["output_sha256"] for x in rows}),
                }
    else:
        summary["error"] = record["error"]
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-python")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-fixtures", action="store_true")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--startup-timeout", type=int, default=360)
    args = parser.parse_args()
    if args.prepare_fixtures:
        prepare_fixtures(args)
        return
    if not args.vllm_python or args.repeats < 1:
        parser.error("--vllm-python and positive --repeats are required")
    cases = json.loads((args.output_dir / "private_prompts.json").read_text(encoding="utf-8"))
    assert all(sha(c["prompt"]) == c["prompt_sha256"] for c in cases)
    variants = [v for v in VARIANTS if not args.variants or v["name"] in args.variants]
    if not variants:
        parser.error("No variants selected")
    summaries = []
    for variant in variants:
        summaries.append(run_variant(args, variant, cases))
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
