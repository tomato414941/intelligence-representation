from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

try:
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import (
        ConditionSummary,
        ContextPredictionResult,
        MemoryInput,
        _build_memory,
        _evaluate_condition,
        _parse_rendered_fact,
        _query_for_action,
        _summarize_by_condition,
    )
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact, RuleBasedPredictor
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import (
        ConditionSummary,
        ContextPredictionResult,
        MemoryInput,
        _build_memory,
        _evaluate_condition,
        _parse_rendered_fact,
        _query_for_action,
        _summarize_by_condition,
    )
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact, RuleBasedPredictor


@dataclass(frozen=True)
class MultiHopCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]


@dataclass(frozen=True)
class MultiHopSummary:
    results: list[ContextPredictionResult]
    condition_summaries: list[ConditionSummary]


class MultiHopContextPredictor:
    def __init__(self, context: list[Observation] | None = None) -> None:
        self.context = list(context or [])
        self.fallback = RuleBasedPredictor()

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if action.type != "find":
            return self.fallback.predict(state, action)

        destination = self._resolve_location_chain(action.object)
        if destination:
            return Fact(subject=action.object, predicate="located_at", object=destination)
        return None

    def _resolve_location_chain(self, subject: str) -> str | None:
        current = subject
        seen = {current}
        found_destination = None

        while True:
            next_location = self._direct_location(current)
            if next_location is None:
                return found_destination
            found_destination = next_location
            if next_location in seen:
                return found_destination
            seen.add(next_location)
            current = next_location

    def _direct_location(self, subject: str) -> str | None:
        for observation in reversed(self.context):
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                return fact.object
        return None


def evaluate_multihop_cases(cases: list[MultiHopCase], *, context_limit: int = 2) -> MultiHopSummary:
    results: list[ContextPredictionResult] = []
    for case in cases:
        memory = _build_memory(case.memory_inputs)
        results.append(_evaluate_condition(case, "no_memory", []))
        direct_context = memory.retrieve(_query_for_action(case.example.action), limit=1)
        results.append(_evaluate_with_multihop_predictor(case, "direct_memory", direct_context))
        multihop_context = _build_multihop_context(memory, case.example.action, context_limit=context_limit)
        results.append(_evaluate_with_multihop_predictor(case, "multi_hop_memory", multihop_context))

    return MultiHopSummary(
        results=results,
        condition_summaries=_summarize_by_condition(results),
    )


def smoke_cases() -> list[MultiHopCase]:
    return [
        MultiHopCase(
            example=ActionConditionedExample(
                id="find_key_on_shelf",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=Fact(subject="鍵", predicate="located_at", object="棚"),
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"]),
                MemoryInput(content="located_at(財布, 机)", tags=["財布", "location"]),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"]),
            ],
        )
    ]


def _evaluate_with_multihop_predictor(
    case: MultiHopCase,
    condition: str,
    context: list[Observation],
) -> ContextPredictionResult:
    expected = case.example.expected_observation
    predicted = MultiHopContextPredictor(context).predict(case.example.state_before, case.example.action)
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


def _build_multihop_context(memory: ObservationMemory, action: Action, *, context_limit: int) -> list[Observation]:
    context: list[Observation] = []
    seen_ids: set[str] = set()
    frontier = [action.object]

    while frontier and len(context) < context_limit:
        current = frontier.pop(0)
        retrieved = memory.retrieve(current, limit=1)
        for observation in retrieved:
            if observation.id in seen_ids:
                continue
            context.append(observation)
            seen_ids.add(observation.id)
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.predicate == "located_at" and fact.object not in frontier:
                frontier.append(fact.object)
            if len(context) >= context_limit:
                break

    return context


def run_demo() -> None:
    summary = evaluate_multihop_cases(smoke_cases())
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
