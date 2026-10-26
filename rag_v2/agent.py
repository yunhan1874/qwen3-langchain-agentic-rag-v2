from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict

from .citations import build_citations, enforce_valid_citations
from .compression import ExtractiveContextCompressor
from .config import RAGConfig
from .generation import AnswerGenerator
from .models import RAGAnswer, RetrievalDetail, RetrievalHit
from .query_rewrite import QueryPlan, Rewriter
from .reranker import Reranker
from .retrieval import HybridRetriever


class AgentState(TypedDict, total=False):
    original_query: str
    explicit_filters: dict[str, Any]
    plan: QueryPlan
    hits: list[RetrievalHit]
    debug_hits: list[RetrievalHit]
    compressed_hits: list[RetrievalHit]
    steps: list[str]
    status: str
    confidence: float
    result: RAGAnswer


@dataclass
class AgenticRAGEngine:
    """Bounded retrieval agent whose actions depend on observed retrieval quality."""

    retriever: HybridRetriever
    reranker: Reranker
    rewriter: Rewriter
    compressor: ExtractiveContextCompressor
    generator: AnswerGenerator
    config: RAGConfig

    def prepare(self, request: str | dict[str, Any]) -> AgentState:
        if isinstance(request, str):
            query = request
            filters: dict[str, Any] = {}
        else:
            query = str(request.get("query", ""))
            filters = dict(request.get("filters") or {})
        query = query.strip()
        if not query:
            raise ValueError("问题不能为空")
        return {
            "original_query": query,
            "explicit_filters": filters,
            "steps": ["prepare_request"],
        }

    def plan_query(self, state: AgentState) -> AgentState:
        # The first retrieval uses the original question; no LLM call is needed here.
        plan = QueryPlan(
            state["original_query"], state["original_query"], state["explicit_filters"]
        )
        return {
            **state,
            "plan": plan,
            "steps": [*state["steps"], "query_plan:direct_retrieval"],
        }

    def agentic_retrieve(self, state: AgentState) -> AgentState:
        plan = state["plan"]
        query = plan.rewritten_query
        best_hits: list[RetrievalHit] = []
        best_score = 0.0
        best_plan = plan
        steps = list(state["steps"])
        seen_queries: set[str] = set()

        for attempt in range(self.config.max_agent_steps):
            if not query or query in seen_queries:
                steps.append("agent_stop:duplicate_query")
                break
            seen_queries.add(query)
            candidates = self.retriever.retrieve(query, plan.filters)
            reranked = self.reranker.rerank(query, candidates, self.config.rerank_k)
            score = reranked[0].rerank_score if reranked else 0.0
            steps.append(
                f"retrieve_attempt:{attempt + 1};hits={len(reranked)};top_score={score:.4f}"
            )
            if score > best_score:
                best_hits, best_score, best_plan = reranked, score, plan
            if reranked and score >= self.config.min_relevance_score:
                steps.append("agent_stop:evidence_sufficient")
                break
            if attempt + 1 >= self.config.max_agent_steps:
                steps.append("agent_stop:max_attempts")
                break
            feedback = "没有候选" if not reranked else f"最高相关度仅 {score:.4f}"
            if attempt == 0:
                rewritten = self.rewriter.rewrite(
                    state["original_query"], state["explicit_filters"]
                )
                plan = QueryPlan(
                    state["original_query"],
                    rewritten.rewritten_query,
                    {**rewritten.filters, **state["explicit_filters"]},
                )
                steps.append(f"agent_action:rewrite_query:{plan.rewritten_query}")
            else:
                query = self.rewriter.retry_query(plan, attempt + 1, feedback)
                plan = QueryPlan(plan.original_query, query, plan.filters)
                steps.append(f"agent_action:retry_query:{query}")
            query = plan.rewritten_query

        accepted = [
            hit for hit in best_hits if hit.rerank_score >= self.config.min_relevance_score
        ][: self.config.final_k]
        status = "evidence_ready" if accepted else "abstained"
        return {
            **state,
            "plan": best_plan if best_hits else plan,
            "hits": accepted,
            "debug_hits": best_hits,
            "status": status,
            "confidence": best_score,
            "steps": steps,
        }

    def compress(self, state: AgentState) -> AgentState:
        if state["status"] == "abstained":
            return {
                **state,
                "compressed_hits": [],
                "steps": [*state["steps"], "skip_compression"],
            }
        compressed = self.compressor.compress(state["plan"].rewritten_query, state["hits"])
        limited: list[RetrievalHit] = []
        used = 0
        for hit in compressed:
            remaining = self.config.max_context_chars - used
            if remaining <= 0:
                break
            text = hit.document.text[:remaining]
            if text != hit.document.text:
                from .models import SourceDocument

                hit = RetrievalHit(
                    document=SourceDocument(hit.document.document_id, text, hit.document.metadata),
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    fusion_score=hit.fusion_score,
                    rerank_score=hit.rerank_score,
                )
            limited.append(hit)
            used += len(text)
        return {
            **state,
            "compressed_hits": limited,
            "steps": [*state["steps"], f"compress_context:chars={used}"],
        }

    def generate(self, state: AgentState) -> AgentState:
        plan = state["plan"]
        accepted_ids = {hit.document.document_id for hit in state.get("hits", [])}
        retrieval_details = [
            RetrievalDetail(
                chunk_id=hit.document.document_id,
                source=str(hit.document.metadata.get("source", "未知来源")),
                title=str(hit.document.metadata.get("title", "")),
                excerpt=hit.document.text[:180].replace("\n", " "),
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                fusion_score=hit.fusion_score,
                rerank_score=hit.rerank_score,
                selected=hit.document.document_id in accepted_ids,
            )
            for hit in state.get("debug_hits", [])
        ]
        if state["status"] == "abstained":
            result = RAGAnswer(
                answer="当前知识库没有检索到足够可靠的证据，无法据此回答。",
                citations=[],
                original_query=plan.original_query,
                rewritten_query=plan.rewritten_query,
                status="abstained",
                agent_steps=state["steps"],
                retrieved_ids=[],
                confidence=state["confidence"],
                retrieval_details=retrieval_details,
            )
            return {**state, "result": result}

        context, citations = build_citations(state["compressed_hits"])
        answer = self.generator.generate(plan.original_query, context)
        answer = enforce_valid_citations(answer, citations)
        result = RAGAnswer(
            answer=answer,
            citations=citations,
            original_query=plan.original_query,
            rewritten_query=plan.rewritten_query,
            status="answered",
            agent_steps=[*state["steps"], "generate_grounded_answer"],
            retrieved_ids=[hit.document.document_id for hit in state["hits"]],
            confidence=state["confidence"],
            retrieval_details=retrieval_details,
        )
        return {**state, "result": result}

    def run(self, query: str, filters: dict[str, Any] | None = None) -> RAGAnswer:
        """Dependency-free path used by tests; production calls the LCEL wrapper below."""
        state = self.prepare({"query": query, "filters": filters or {}})
        state = self.plan_query(state)
        state = self.agentic_retrieve(state)
        state = self.compress(state)
        return self.generate(state)["result"]


class LangChainAgenticRAG:
    """Public production facade: the complete request lifecycle is a LangChain LCEL graph."""

    def __init__(self, engine: AgenticRAGEngine) -> None:
        try:
            from langchain_core.runnables import RunnableLambda
        except ImportError as exc:
            raise RuntimeError("请先安装 langchain-core") from exc
        self.engine = engine
        self.chain = (
            RunnableLambda(engine.prepare, name="prepare_request")
            | RunnableLambda(engine.plan_query, name="prepare_query")
            | RunnableLambda(engine.agentic_retrieve, name="agentic_hybrid_retrieval")
            | RunnableLambda(engine.compress, name="context_compression")
            | RunnableLambda(engine.generate, name="cited_grounded_generation")
            | RunnableLambda(lambda state: state["result"], name="return_answer")
        ).with_config(run_name="qwen3_medical_agentic_rag_v2")

    def invoke(self, query: str, filters: dict[str, Any] | None = None) -> RAGAnswer:
        return self.chain.invoke({"query": query, "filters": filters or {}})
