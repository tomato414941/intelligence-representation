from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

try:
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import MemoryInput, _build_memory, _parse_rendered_fact
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import MemoryInput, _build_memory, _parse_rendered_fact
    from experiments.observation_memory import Observation, ObservationMemory
    from experiments.predictor_interface import Action, Fact


PredictionState = Literal["unsupported", "resolved", "ambiguous"]


@dataclass(frozen=True)
class AmbiguousMultiHopCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]


@dataclass(frozen=True)
class CandidatePrediction:
    state: PredictionState
    candidates: list[Fact]
    context_observation_ids: list[str]


@dataclass(frozen=True)
class AmbiguousPredictionResult:
    case_name: str
    prediction_state: PredictionState
    candidate_facts: list[Fact]
    expected_candidates: list[Fact]
    correct: bool
    context_observation_ids: list[str]


@dataclass(frozen=True)
class AmbiguousPredictionSummary:
    results: list[AmbiguousPredictionResult]
    accuracy: float
    ambiguous_rate: float


class AmbiguousMultiHopPredictor:
    def __init__(self, context: list[Observation] | None = None) -> None:
        self.context = list(context or [])

    def predict(self, action: Action) -> CandidatePrediction:
        if action.type != "find":
            return CandidatePrediction(state="unsupported", candidates=[], context_observation_ids=[])

        destinations = self._resolve_candidates(action.object)
        candidates = [Fact(subject=action.object, predicate="located_at", object=destination) for destination in destinations]
        if not candidates:
            state: PredictionState = "unsupported"
        elif len(candidates) == 1:
            state = "resolved"
        else:
            state = "ambiguous"

        return CandidatePrediction(
            state=state,
            candidates=candidates,
            context_observation_ids=[observation.id for observation in self.context],
        )

    def _resolve_candidates(self, subject: str) -> list[str]:
        direct_locations = self._direct_locations(subject)
        if not direct_locations:
            return []

        destinations: list[str] = []
        for location in direct_locations:
            next_locations = self._direct_locations(location)
            if not next_locations:
                destinations.append(location)
            else:
                destinations.extend(next_locations)

        return sorted(set(destinations))

    def _direct_locations(self, subject: str) -> list[str]:
        locations = []
        for observation in self.context:
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                locations.append(fact.object)
        return locations


def evaluate_ambiguous_multihop_cases(cases: list[AmbiguousMultiHopCase]) -> AmbiguousPredictionSummary:
    results = []
    for case in cases:
        memory = _build_memory(case.memory_inputs)
        context = _build_full_location_context(memory)
        prediction = AmbiguousMultiHopPredictor(context).predict(case.example.action)
        expected_candidates = case.example.expected_state_after
        results.append(
            AmbiguousPredictionResult(
                case_name=case.example.id,
                prediction_state=prediction.state,
                candidate_facts=prediction.candidates,
                expected_candidates=expected_candidates,
                correct=_same_fact_set(prediction.candidates, expected_candidates),
                context_observation_ids=prediction.context_observation_ids,
            )
        )

    if not results:
        return AmbiguousPredictionSummary(results=[], accuracy=0.0, ambiguous_rate=0.0)

    return AmbiguousPredictionSummary(
        results=results,
        accuracy=sum(1 for result in results if result.correct) / len(results),
        ambiguous_rate=sum(1 for result in results if result.prediction_state == "ambiguous") / len(results),
    )


def smoke_cases() -> list[AmbiguousMultiHopCase]:
    return [
        AmbiguousMultiHopCase(
            example=ActionConditionedExample(
                id="find_key_ambiguous_container",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=None,
                expected_state_after=[
                    Fact(subject="鍵", predicate="located_at", object="机"),
                    Fact(subject="鍵", predicate="located_at", object="棚"),
                ],
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"]),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"]),
                MemoryInput(content="located_at(箱, 机)", tags=["箱", "location"]),
            ],
        )
    ]


def _build_full_location_context(memory: ObservationMemory) -> list[Observation]:
    return [observation for observation in memory.observations if _parse_rendered_fact(observation.content)]


def _same_fact_set(left: list[Fact], right: list[Fact]) -> bool:
    return {fact.key() for fact in left} == {fact.key() for fact in right}


def run_demo() -> None:
    summary = evaluate_ambiguous_multihop_cases(smoke_cases())
    print(f"accuracy={summary.accuracy:.2f} ambiguous_rate={summary.ambiguous_rate:.2f}")
    for result in summary.results:
        candidates = ", ".join(fact.render() for fact in result.candidate_facts) or "unsupported"
        expected = ", ".join(fact.render() for fact in result.expected_candidates) or "none"
        print(
            f"{result.case_name}: state={result.prediction_state} "
            f"candidates=[{candidates}] expected=[{expected}] "
            f"correct={result.correct} context={result.context_observation_ids}"
        )


if __name__ == "__main__":
    run_demo()
