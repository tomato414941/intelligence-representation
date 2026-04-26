from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys

try:
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor


@dataclass(frozen=True)
class MemoryInput:
    content: str
    tags: list[str] = field(default_factory=list)
    timestamp: str = "t0"


@dataclass(frozen=True)
class ObservationAssistedCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]


@dataclass(frozen=True)
class ContextPredictionResult:
    case_name: str
    condition: str
    predicted_fact: Fact | None
    expected_fact: Fact | None
    correct: bool
    context_size: int
    retrieved_observation_ids: list[str]


@dataclass(frozen=True)
class ConditionSummary:
    condition: str
    accuracy: float
    average_context_size: float


@dataclass(frozen=True)
class ObservationAssistedSummary:
    results: list[ContextPredictionResult]
    condition_summaries: list[ConditionSummary]


class ContextFactPredictor:
    def __init__(self, context: list[Observation] | None = None, fallback: Predictor | None = None) -> None:
        self.context = list(context or [])
        self.fallback = fallback or RuleBasedPredictor()

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if action.type == "find":
            fact = self._find_location_fact(action.object)
            if fact:
                return fact
        return self.fallback.predict(state, action)

    def _find_location_fact(self, subject: str) -> Fact | None:
        for observation in reversed(self.context):
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                return fact
        return None


def evaluate_observation_assisted_cases(
    cases: list[ObservationAssistedCase],
    *,
    context_limit: int = 2,
) -> ObservationAssistedSummary:
    results: list[ContextPredictionResult] = []
    for case in cases:
        memory = _build_memory(case.memory_inputs)
        results.append(_evaluate_condition(case, "no_memory", []))
        recent_context = memory.observations[-context_limit:]
        results.append(_evaluate_condition(case, "recent_memory", recent_context))
        retrieved_context = memory.retrieve(_query_for_action(case.example.action), limit=context_limit)
        results.append(_evaluate_condition(case, "retrieved_memory", retrieved_context))

    return ObservationAssistedSummary(
        results=results,
        condition_summaries=_summarize_by_condition(results),
    )


def smoke_cases() -> list[ObservationAssistedCase]:
    return [
        ObservationAssistedCase(
            example=ActionConditionedExample(
                id="find_key_location",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=Fact(subject="鍵", predicate="located_at", object="棚"),
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 棚)", tags=["鍵", "location"]),
                MemoryInput(content="located_at(財布, 机)", tags=["財布", "location"]),
                MemoryInput(content="located_at(本, 図書館)", tags=["本", "location"]),
            ],
        ),
        ObservationAssistedCase(
            example=ActionConditionedExample(
                id="place_book_library",
                state_before=[Fact(subject="佐藤", predicate="has", object="本")],
                action=Action(type="place", actor="佐藤", object="本", target="図書館"),
                expected_observation=Fact(subject="本", predicate="located_at", object="図書館"),
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 棚)", tags=["鍵", "location"]),
            ],
        ),
    ]


def _evaluate_condition(
    case: ObservationAssistedCase,
    condition: str,
    context: list[Observation],
) -> ContextPredictionResult:
    expected = case.example.expected_observation
    predicted = ContextFactPredictor(context).predict(case.example.state_before, case.example.action)
    correct = predicted is None if expected is None else predicted is not None and predicted.key() == expected.key()
    return ContextPredictionResult(
        case_name=case.example.id,
        condition=condition,
        predicted_fact=predicted,
        expected_fact=expected,
        correct=correct,
        context_size=len(context),
        retrieved_observation_ids=[observation.id for observation in context],
    )


def _build_memory(memory_inputs: list[MemoryInput]) -> ObservationMemory:
    memory = ObservationMemory()
    for memory_input in memory_inputs:
        memory.add(
            memory_input.content,
            timestamp=memory_input.timestamp,
            tags=memory_input.tags,
            source="fixture",
        )
    return memory


def _query_for_action(action: Action) -> str:
    return f"{action.type} {action.object} {action.target}"


def _parse_rendered_fact(content: str) -> Fact | None:
    match = re.fullmatch(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\(([^,]+),\s*([^)]+)\)\s*", content)
    if not match:
        return None
    return Fact(subject=match.group(2).strip(), predicate=match.group(1).strip(), object=match.group(3).strip())


def _summarize_by_condition(results: list[ContextPredictionResult]) -> list[ConditionSummary]:
    summaries = []
    for condition in sorted({result.condition for result in results}):
        condition_results = [result for result in results if result.condition == condition]
        summaries.append(
            ConditionSummary(
                condition=condition,
                accuracy=sum(1 for result in condition_results if result.correct) / len(condition_results),
                average_context_size=sum(result.context_size for result in condition_results) / len(condition_results),
            )
        )
    return summaries


def run_demo() -> None:
    summary = evaluate_observation_assisted_cases(smoke_cases())
    for condition_summary in summary.condition_summaries:
        print(
            f"{condition_summary.condition}: "
            f"accuracy={condition_summary.accuracy:.2f} "
            f"avg_context={condition_summary.average_context_size:.1f}"
        )
    for result in summary.results:
        predicted = result.predicted_fact.render() if result.predicted_fact else "unsupported"
        expected = result.expected_fact.render() if result.expected_fact else "unsupported"
        print(
            f"{result.case_name}/{result.condition}: "
            f"predicted={predicted} expected={expected} "
            f"correct={result.correct} context={result.retrieved_observation_ids}"
        )


if __name__ == "__main__":
    run_demo()
