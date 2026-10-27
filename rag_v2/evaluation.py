from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgenticRAGEngine, LangChainAgenticRAG
from .citations import has_valid_citation
from .text import split_sentences, tokenize


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_ids: list[str]
    filters: dict[str, Any]


@dataclass(frozen=True)
class EvaluationReport:
    recall_at_k: float
    lexical_support_proxy: float
    exact_sentence_support: float
    case_count: int
    k: int
    answered_count: int
    abstention_rate: float
    citation_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            f"recall@{self.k}": self.recall_at_k,
            "faithfulness": None,
            "faithfulness_method": "not_evaluated",
            "lexical_support_proxy": self.lexical_support_proxy,
            "exact_sentence_support": self.exact_sentence_support,
            "case_count": self.case_count,
            "answered_count": self.answered_count,
            "abstention_rate": self.abstention_rate,
            "citation_rate": self.citation_rate,
        }


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & set(relevant_ids)) / len(set(relevant_ids))


def lexical_support_score(answer: str, contexts: list[str], threshold: float = 0.35) -> float:
    """Lexical overlap proxy only; it cannot detect entailment or contradiction."""
    answer = re.sub(r"\[\d+\]", "", answer)
    claims = [claim for claim in split_sentences(answer) if tokenize(claim)]
    if not claims:
        return 0.0
    context_tokens = set(tokenize("\n".join(contexts)))
    supported = 0
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        coverage = len(claim_tokens & context_tokens) / (len(claim_tokens) or 1)
        supported += coverage >= threshold
    return supported / len(claims)


def faithfulness_score(answer: str, contexts: list[str], threshold: float = 0.35) -> float:
    """Legacy alias for lexical_support_score, NOT a semantic faithfulness judge."""
    return lexical_support_score(answer, contexts, threshold)


def exact_sentence_support_score(answer: str, contexts: list[str]) -> float:
    """Conservative verbatim-sentence metric; paraphrases may score zero."""
    def sentences(text: str) -> list[str]:
        text = re.sub(r"\[\d+\]", "", text)
        return [
            re.sub(r"\s+", "", part).strip("。！？!?；;，,：:.")
            for part in split_sentences(text) if tokenize(part)
        ]
    claims = sentences(answer)
    evidence = {sentence for context in contexts for sentence in sentences(context)}
    return sum(claim in evidence for claim in claims) / len(claims) if claims else 0.0


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cases.append(
                EvaluationCase(
                    query=row["query"],
                    relevant_ids=list(row["relevant_ids"]),
                    filters=dict(row.get("filters") or {}),
                )
            )
    return cases


class RAGEvaluator:
    def __init__(self, engine: AgenticRAGEngine) -> None:
        self.engine = engine

    def evaluate(self, cases: list[EvaluationCase], k: int = 3) -> EvaluationReport:
        if k < 1:
            raise ValueError("k 必须为正整数")
        recalls: list[float] = []
        lexical_scores: list[float] = []
        exact_scores: list[float] = []
        cited = abstained = 0
        rag = LangChainAgenticRAG(self.engine)
        for case in cases:
            answer = rag.invoke(case.query, case.filters)
            recalls.append(recall_at_k(answer.retrieved_ids, case.relevant_ids, k))
            abstained += answer.status == "abstained"
            if answer.status == "answered" and answer.answer.strip():
                contexts = [citation.evidence or citation.excerpt for citation in answer.citations]
                lexical_scores.append(lexical_support_score(answer.answer, contexts))
                exact_scores.append(exact_sentence_support_score(answer.answer, contexts))
                cited += has_valid_citation(answer.answer, answer.citations)
        count = len(cases)
        answered = len(lexical_scores)
        return EvaluationReport(
            recall_at_k=sum(recalls) / count if count else 0.0,
            lexical_support_proxy=sum(lexical_scores) / answered if answered else 0.0,
            exact_sentence_support=sum(exact_scores) / answered if answered else 0.0,
            case_count=count,
            k=k,
            answered_count=answered,
            abstention_rate=abstained / count if count else 0.0,
            citation_rate=cited / answered if answered else 0.0,
        )
