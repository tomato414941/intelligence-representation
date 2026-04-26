from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

try:
    from experiments.ngram_observation_memory import NgramObservationMemory
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.ngram_observation_memory import NgramObservationMemory


@dataclass(frozen=True)
class EvaluationObservation:
    id: str
    content: str
    tags: list[str]


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    observations: list[EvaluationObservation]
    query: str
    expected_relevant_ids: list[str]
    k: int = 5


@dataclass(frozen=True)
class EvaluationResult:
    case_name: str
    retrieved_ids: list[str]
    expected_relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class EvaluationSummary:
    results: list[EvaluationResult]
    mean_precision_at_k: float
    mean_recall_at_k: float


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        observations: list[EvaluationObservation],
        *,
        limit: int,
    ) -> list[EvaluationObservation]:
        ...


class NgramRetrieverAdapter:
    def retrieve(
        self,
        query: str,
        observations: list[EvaluationObservation],
        *,
        limit: int,
    ) -> list[EvaluationObservation]:
        memory = NgramObservationMemory()
        generated_to_original: dict[str, EvaluationObservation] = {}
        for observation in observations:
            generated = memory.add(observation.content, tags=observation.tags)
            generated_to_original[generated.id] = observation

        retrieved = memory.retrieve(query, limit=limit)
        return [generated_to_original[observation.id] for observation in retrieved]


def evaluate_case(case: EvaluationCase, retriever: Retriever) -> EvaluationResult:
    retrieved = retriever.retrieve(case.query, case.observations, limit=case.k)
    retrieved_ids = [observation.id for observation in retrieved]
    expected_ids = set(case.expected_relevant_ids)
    retrieved_id_set = set(retrieved_ids)

    if retrieved_ids:
        precision = len(retrieved_id_set & expected_ids) / len(retrieved_ids)
    else:
        precision = 0.0

    if expected_ids:
        recall = len(retrieved_id_set & expected_ids) / len(expected_ids)
    else:
        recall = 1.0

    return EvaluationResult(
        case_name=case.name,
        retrieved_ids=retrieved_ids,
        expected_relevant_ids=case.expected_relevant_ids,
        precision_at_k=precision,
        recall_at_k=recall,
    )


def evaluate_cases(cases: list[EvaluationCase], retriever: Retriever) -> EvaluationSummary:
    results = [evaluate_case(case, retriever) for case in cases]
    if not results:
        return EvaluationSummary(results=[], mean_precision_at_k=0.0, mean_recall_at_k=0.0)

    return EvaluationSummary(
        results=results,
        mean_precision_at_k=sum(result.precision_at_k for result in results) / len(results),
        mean_recall_at_k=sum(result.recall_at_k for result in results) / len(results),
    )


def smoke_cases() -> list[EvaluationCase]:
    observations = [
        EvaluationObservation(
            id="obs_1",
            content="Transformerは系列モデルというより関係計算器に近い",
            tags=["transformer", "関係"],
        ),
        EvaluationObservation(
            id="obs_2",
            content="Attentionはトークン間の重み付き関係を作る",
            tags=["attention", "関係"],
        ),
        EvaluationObservation(
            id="obs_3",
            content="State Memoryは真理DBではなくキャッシュとして扱う",
            tags=["state-memory", "cache"],
        ),
    ]
    return [
        EvaluationCase(
            name="transformer_relation",
            observations=observations,
            query="Transformer 関係",
            expected_relevant_ids=["obs_1", "obs_2"],
            k=2,
        ),
        EvaluationCase(
            name="state_memory_cache",
            observations=observations,
            query="キャッシュ",
            expected_relevant_ids=["obs_3"],
            k=2,
        ),
        EvaluationCase(
            name="attention_tag",
            observations=observations,
            query="attention",
            expected_relevant_ids=["obs_2"],
            k=1,
        ),
    ]


def run_demo() -> None:
    summary = evaluate_cases(smoke_cases(), NgramRetrieverAdapter())
    for result in summary.results:
        print(
            f"{result.case_name}: retrieved={result.retrieved_ids} "
            f"precision={result.precision_at_k:.2f} recall={result.recall_at_k:.2f}"
        )
    print(
        f"mean_precision={summary.mean_precision_at_k:.2f} "
        f"mean_recall={summary.mean_recall_at_k:.2f}"
    )


if __name__ == "__main__":
    run_demo()
