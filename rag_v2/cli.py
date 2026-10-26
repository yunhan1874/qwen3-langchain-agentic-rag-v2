from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import AgenticRAGEngine, LangChainAgenticRAG
from .compression import ExtractiveContextCompressor
from .config import RAGConfig
from .embeddings import SentenceTransformerEmbedder
from .evaluation import RAGEvaluator, load_evaluation_cases
from .generation import LangChainGroundedGenerator
from .query_rewrite import LLMQueryRewriter
from .reranker import CrossEncoderReranker
from .retrieval import ChromaDenseIndex, HybridRetriever
from .semantic_chunking import SemanticChunker, load_chunks, load_source_documents, save_chunks


def _build_llm(config: RAGConfig) -> Any:
    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("请先执行 pip install -r requirements.txt") from exc

    print("加载本地模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, device_map="auto", torch_dtype="auto"
    )
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.answer_max_new_tokens,
        temperature=0.1,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=text_pipeline)


def build_engine(config: RAGConfig, rebuild_index: bool = False) -> AgenticRAGEngine:
    config.validate()
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    chunks_path = config.index_dir / "chunks.jsonl"
    if rebuild_index or not chunks_path.exists():
        print("执行 Semantic Chunking...")
        source_documents = load_source_documents(config.docs_path)
        chunker = SemanticChunker(
            embedder,
            breakpoint_percentile=config.semantic_breakpoint_percentile,
            min_chunk_chars=config.min_chunk_chars,
            max_chunk_chars=config.max_chunk_chars,
        )
        chunks = chunker.split_documents(source_documents)
        save_chunks(chunks, chunks_path)
    else:
        chunks = load_chunks(chunks_path)

    print(f"加载混合索引，共 {len(chunks)} 个 semantic chunks...")
    dense_index = ChromaDenseIndex(
        chunks,
        embedder,
        persist_directory=config.index_dir / "chroma",
        collection_name=config.collection_name,
        rebuild=rebuild_index,
    )
    retriever = HybridRetriever(
        chunks,
        dense_index,
        dense_k=config.dense_k,
        bm25_k=config.bm25_k,
        fused_k=config.fused_k,
        rrf_constant=config.rrf_constant,
    )
    llm = _build_llm(config)
    return AgenticRAGEngine(
        retriever=retriever,
        reranker=CrossEncoderReranker(config.reranker_model),
        rewriter=LLMQueryRewriter(
            llm.bind(pipeline_kwargs={"max_new_tokens": config.rewrite_max_new_tokens})
        ),
        compressor=ExtractiveContextCompressor(config.compression_max_chars),
        generator=LangChainGroundedGenerator(llm),
        config=config,
    )


def _parse_filter(values: list[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"过滤条件必须是 key=value：{value}")
        key, raw = value.split("=", 1)
        raw = raw.strip()
        if raw.isdigit():
            parsed: Any = int(raw)
        elif raw.lower() in {"true", "false"}:
            parsed = raw.lower() == "true"
        else:
            parsed = raw
        filters[key.strip()] = parsed
    return filters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 医疗 LangChain Agentic RAG v2")
    parser.add_argument("--docs", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--query", help="单次查询；不传则进入交互模式")
    parser.add_argument("--filter", action="append", default=[], help="Metadata 过滤：key=value")
    parser.add_argument("--evaluate", type=Path, help="评估 JSONL")
    parser.add_argument("--eval-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RAGConfig()
    if args.docs:
        config = replace(config, docs_path=args.docs)
    if args.model:
        config = replace(config, model_path=args.model)
    engine = build_engine(config, rebuild_index=args.rebuild_index)

    if args.evaluate:
        report = RAGEvaluator(engine).evaluate(load_evaluation_cases(args.evaluate), args.eval_k)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return

    rag = LangChainAgenticRAG(engine)
    filters = _parse_filter(args.filter)
    if args.query:
        print(rag.invoke(args.query, filters).render())
        return

    while True:
        query = input("\n请输入你的问题（输入 exit 退出）：\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        result = rag.invoke(query, filters)
        print("\n=== 回答 ===")
        print(result.render())
        print("\n=== Agent 轨迹 ===")
        print(" -> ".join(result.agent_steps))


if __name__ == "__main__":
    main()
