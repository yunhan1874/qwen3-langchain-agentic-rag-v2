from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgenticRAGEngine
from .text import split_sentences, tokenize


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_ids: list[str]
    filters: dict[str, Any]


@dataclass(frozen=True)
class EvaluationReport:
    recall_at_k: float
    faithfulness: float
    case_count: int
    k: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            f"recall@{self.k}": self.recall_at_k,
            "faithfulness": self.faithfulness,
            "case_count": self.case_count,
        }


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & set(relevant_ids)) / len(set(relevant_ids))


def faithfulness_score(answer: str, contexts: list[str], threshold: float = 0.35) -> float:
    """Measure the share of answer claims lexically supported by cited contexts."""
    answer = re.sub(r"\[\d+\]", "", answer)
    claims = [claim for claim in split_sentences(answer) if tokenize(claim)]
    if not claims:
        return 1.0
    context_tokens = set(tokenize("\n".join(contexts)))
    supported = 0
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        coverage = len(claim_tokens & context_tokens) / (len(claim_tokens) or 1)
        supported += coverage >= threshold
    return supported / len(claims)


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
        recalls: list[float] = []
        faithfulness_scores: list[float] = []
        for case in cases:
            answer = self.engine.run(case.query, case.filters)
            recalls.append(recall_at_k(answer.retrieved_ids, case.relevant_ids, k))
            faithfulness_scores.append(
                faithfulness_score(answer.answer, [citation.excerpt for citation in answer.citations])
            )
        count = len(cases)
        return EvaluationReport(
            recall_at_k=sum(recalls) / count if count else 0.0,
            faithfulness=sum(faithfulness_scores) / count if count else 0.0,
            case_count=count,
            k=k,
        )
