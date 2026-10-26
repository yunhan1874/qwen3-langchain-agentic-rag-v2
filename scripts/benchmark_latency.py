"""Time the complete Gradio request, including queue and network overhead."""
from __future__ import annotations

import argparse
import json
import statistics
import time

from gradio_client import Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:17860")
    parser.add_argument("--query", default="高血压患者日常如何管理血压？")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("runs must be positive; warmup must be non-negative")
    client = Client(args.url, verbose=False)
    durations = []
    for index in range(args.warmup + args.runs):
        started = time.perf_counter()
        result = client.predict(
            args.query, "真实模式（Qwen + BGE + Chroma）", "{}", api_name="/handle_request"
        )
        elapsed = time.perf_counter() - started
        is_warmup = index < args.warmup
        print(json.dumps({
            "warmup": is_warmup, "seconds": round(elapsed, 3), "status": result[4],
            "citations": len(result[1]), "steps": result[3],
        }, ensure_ascii=False), flush=True)
        if "`error`" in result[4]:
            raise RuntimeError("Gradio returned an error; timings are not a successful benchmark")
        if not is_warmup:
            durations.append(elapsed)
    print(json.dumps({
        "runs": len(durations), "median_seconds": round(statistics.median(durations), 3),
        "mean_seconds": round(statistics.mean(durations), 3),
    }), flush=True)


if __name__ == "__main__":
    main()
