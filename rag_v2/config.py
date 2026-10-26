from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RAGConfig:
    """Runtime settings; model and source data remain the same as the original project."""

    model_path: str = field(
        default_factory=lambda: os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen3-1.7B")
    )
    docs_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAG_DOCS_PATH", str(PROJECT_ROOT / "data" / "docs.txt"))
        )
    )
    index_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "rag_v2_index"))
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    collection_name: str = "qwen3_medical_semantic_chunks"
    max_length: int = 2048

    semantic_breakpoint_percentile: float = 25.0
    min_chunk_chars: int = 120
    max_chunk_chars: int = 900

    dense_k: int = 12
    bm25_k: int = 12
    fused_k: int = 12
    rerank_k: int = 5
    final_k: int = 3
    rrf_constant: int = 60
    min_relevance_score: float = 0.28

    compression_max_chars: int = 650
    max_context_chars: int = 1800
    max_agent_steps: int = 3

    def validate(self) -> None:
        positive = {
            "max_length": self.max_length,
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "dense_k": self.dense_k,
            "bm25_k": self.bm25_k,
            "fused_k": self.fused_k,
            "rerank_k": self.rerank_k,
            "final_k": self.final_k,
            "rrf_constant": self.rrf_constant,
            "compression_max_chars": self.compression_max_chars,
            "max_context_chars": self.max_context_chars,
            "max_agent_steps": self.max_agent_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"配置必须为正数: {', '.join(invalid)}")
        if self.min_chunk_chars > self.max_chunk_chars:
            raise ValueError("min_chunk_chars 不能大于 max_chunk_chars")
        if self.final_k > self.rerank_k or self.rerank_k > self.fused_k:
            raise ValueError("必须满足 final_k <= rerank_k <= fused_k")
        if not 0 <= self.min_relevance_score <= 1:
            raise ValueError("min_relevance_score 必须位于 [0, 1]")
        if not 0 <= self.semantic_breakpoint_percentile <= 100:
            raise ValueError("semantic_breakpoint_percentile 必须位于 [0, 100]")
