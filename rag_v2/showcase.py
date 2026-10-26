from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

from .agent import AgenticRAGEngine
from .compression import ExtractiveContextCompressor
from .config import RAGConfig
from .embeddings import HashingEmbedder
from .generation import ExtractiveGenerator
from .models import RAGAnswer, SourceDocument
from .query_rewrite import RuleBasedRewriter
from .reranker import TokenOverlapReranker
from .retrieval import HybridRetriever, InMemoryDenseIndex


DEMO_NOTICE = (
    "演示模式使用仓库内的合成语料与轻量替代模型，只用于展示 Agentic RAG "
    "流程和界面，不代表真实医疗知识库结果，也不构成医疗建议。"
)


def demo_documents() -> list[SourceDocument]:
    """Small synthetic corpus; it contains no source medical records."""
    return [
        SourceDocument(
            "demo-cardio-001",
            (
                "演示资料：血压管理记录可以包含规律测量、记录趋势和复诊时向专业人员"
                "展示记录。任何治疗调整都应由具备资质的医疗专业人员结合个体情况决定。"
            ),
            {
                "source": "synthetic_demo_corpus",
                "title": "血压记录演示条目",
                "department": "心内科",
                "category": "健康管理",
                "year": 2026,
                "document_type": "synthetic_demo",
            },
        ),
        SourceDocument(
            "demo-endo-001",
            (
                "演示资料：血糖随访可以记录测量时间、饮食和活动等背景信息，便于就诊时"
                "沟通。出现异常或不适时，应咨询医疗专业人员。"
            ),
            {
                "source": "synthetic_demo_corpus",
                "title": "血糖随访演示条目",
                "department": "内分泌科",
                "category": "健康管理",
                "year": 2026,
                "document_type": "synthetic_demo",
            },
        ),
        SourceDocument(
            "demo-respiratory-001",
            (
                "演示资料：呼吸道症状记录可包括持续时间、变化趋势及伴随不适，供就诊沟通"
                "参考。该演示条目不用于诊断或推荐药物。"
            ),
            {
                "source": "synthetic_demo_corpus",
                "title": "症状记录演示条目",
                "department": "呼吸科",
                "category": "就诊准备",
                "year": 2025,
                "document_type": "synthetic_demo",
            },
        ),
        SourceDocument(
            "demo-safety-001",
            (
                "演示资料：如果用户描述紧急或快速恶化的健康状况，系统应停止一般性回答，"
                "提示尽快联系当地急救服务或合格医疗专业人员。"
            ),
            {
                "source": "synthetic_demo_corpus",
                "title": "安全边界演示条目",
                "department": "急诊科",
                "category": "安全提示",
                "year": 2026,
                "document_type": "synthetic_demo",
            },
        ),
        SourceDocument(
            "demo-system-001",
            (
                "演示资料：检索系统先融合关键词与向量召回结果，再重排候选，并只使用通过"
                "相关度阈值的证据生成带编号引用的回答。"
            ),
            {
                "source": "synthetic_demo_corpus",
                "title": "RAG 流程演示条目",
                "department": "工程",
                "category": "系统说明",
                "year": 2026,
                "document_type": "synthetic_demo",
            },
        ),
    ]


def build_demo_engine() -> AgenticRAGEngine:
    documents = demo_documents()
    dense_index = InMemoryDenseIndex(documents, HashingEmbedder())
    config = replace(
        RAGConfig(),
        dense_k=5,
        bm25_k=5,
        fused_k=5,
        rerank_k=5,
        final_k=3,
        # The lightweight token-overlap reranker still receives a small RRF prior.
        # Keep the demo gate above that prior so out-of-domain questions abstain.
        min_relevance_score=0.25,
        compression_max_chars=260,
        max_context_chars=720,
    )
    return AgenticRAGEngine(
        retriever=HybridRetriever(
            documents,
            dense_index,
            dense_k=config.dense_k,
            bm25_k=config.bm25_k,
            fused_k=config.fused_k,
            rrf_constant=config.rrf_constant,
        ),
        reranker=TokenOverlapReranker(),
        rewriter=RuleBasedRewriter(),
        compressor=ExtractiveContextCompressor(config.compression_max_chars),
        generator=ExtractiveGenerator(),
        config=config,
    )


def parse_metadata_filter(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        filters = value
    else:
        try:
            filters = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Metadata Filter 必须是合法 JSON 对象") from exc
    if not isinstance(filters, dict):
        raise ValueError("Metadata Filter 必须是 JSON 对象")
    for key, item in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata Filter 的字段名必须是非空字符串")
        if not isinstance(item, (str, int, float, bool, list)) and item is not None:
            raise ValueError(f"Metadata Filter 字段 {key} 使用了不支持的值类型")
    return filters


def result_payload(result: RAGAnswer, demo_mode: bool) -> dict[str, Any]:
    citations = [asdict(citation) for citation in result.citations]
    scores = [
        {
            "chunk_id": detail.chunk_id,
            "dense": round(detail.dense_score, 4),
            "bm25": round(detail.bm25_score, 4),
            "rrf": round(detail.fusion_score, 4),
            "rerank": round(detail.rerank_score, 4),
            "selected": detail.selected,
            "source": detail.source,
        }
        for detail in result.retrieval_details
    ]
    prefix = f"> ⚠️ **{DEMO_NOTICE}**\n\n" if demo_mode else ""
    return {
        "answer_markdown": prefix + result.render(),
        "citations": citations,
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "agent_steps": result.agent_steps,
        "status": result.status,
        "confidence": round(result.confidence, 4),
        "scores": scores,
    }


def run_demo(query: str, metadata_filter: str | dict[str, Any] | None = None) -> dict[str, Any]:
    filters = parse_metadata_filter(metadata_filter)
    return result_payload(build_demo_engine().run(query, filters), demo_mode=True)
