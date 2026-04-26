from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

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


@dataclass(frozen=True)
class TemporalMultiHopCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]


@dataclass(frozen=True)
class TemporalPrediction:
    fact: Fact | None
    provenance_observation_ids: list[str]
    superseded_observation_ids: list[str]


@dataclass(frozen=True)
class TemporalPredictionResult:
    case_name: str
    predicted_fact: Fact | None
    expected_fact: Fact | None
    correct: bool
    provenance_observation_ids: list[str]
    superseded_observation_ids: list[str]


@dataclass(frozen=True)
class TemporalPredictionSummary:
    results: list[TemporalPredictionResult]
    accuracy: float


class TemporalMultiHopPredictor:
    def __init__(self, context: list[Observation] | None = None) -> None:
        self.context = list(context or [])

    def predict(self, action: Action) -> TemporalPrediction:
        if action.type != "find":
            return TemporalPrediction(fact=None, provenance_observation_ids=[], superseded_observation_ids=[])

        current = action.object
        seen = {current}
        provenance: list[str] = []
        superseded: list[str] = []
        destination = None

        while True:
            selected, replaced = self._latest_location_observation(current)
            superseded.extend(observation.id for observation in replaced)
            if selected is None:
                break

            fact = _parse_rendered_fact(selected.content)
            if fact is None:
                break

            provenance.append(selected.id)
            destination = fact.object
            if destination in seen:
                break
            seen.add(destination)
            current = destination

        if destination is None:
            return TemporalPrediction(
                fact=None,
                provenance_observation_ids=provenance,
                superseded_observation_ids=_unique(superseded),
            )

        return TemporalPrediction(
            fact=Fact(subject=action.object, predicate="located_at", object=destination),
            provenance_observation_ids=provenance,
            superseded_observation_ids=_unique(superseded),
        )

    def _latest_location_observation(self, subject: str) -> tuple[Observation | None, list[Observation]]:
        matching = []
        for observation in self.context:
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                matching.append(observation)

        if not matching:
            return None, []

        matching.sort(key=lambda observation: observation.timestamp)
        return matching[-1], matching[:-1]


def evaluate_temporal_multihop_cases(cases: list[TemporalMultiHopCase]) -> TemporalPredictionSummary:
    results = []
    for case in cases:
        memory = _build_memory(case.memory_inputs)
        context = _build_full_location_context(memory)
        prediction = TemporalMultiHopPredictor(context).predict(case.example.action)
        expected = case.example.expected_observation
        results.append(
            TemporalPredictionResult(
                case_name=case.example.id,
                predicted_fact=prediction.fact,
                expected_fact=expected,
                correct=prediction.fact is not None and expected is not None and prediction.fact.key() == expected.key(),
                provenance_observation_ids=prediction.provenance_observation_ids,
                superseded_observation_ids=prediction.superseded_observation_ids,
            )
        )

    if not results:
        return TemporalPredictionSummary(results=[], accuracy=0.0)

    return TemporalPredictionSummary(
        results=results,
        accuracy=sum(1 for result in results if result.correct) / len(results),
    )


def smoke_cases() -> list[TemporalMultiHopCase]:
    return [
        TemporalMultiHopCase(
            example=ActionConditionedExample(
                id="find_key_after_container_moves",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=Fact(subject="鍵", predicate="located_at", object="机"),
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"], timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"], timestamp="t2"),
                MemoryInput(content="located_at(箱, 机)", tags=["箱", "location"], timestamp="t3"),
            ],
        )
    ]


def _build_full_location_context(memory: ObservationMemory) -> list[Observation]:
    return [observation for observation in memory.observations if _parse_rendered_fact(observation.content)]


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def run_demo() -> None:
    summary = evaluate_temporal_multihop_cases(smoke_cases())
    print(f"accuracy={summary.accuracy:.2f}")
    for result in summary.results:
        predicted = result.predicted_fact.render() if result.predicted_fact else "unsupported"
        expected = result.expected_fact.render() if result.expected_fact else "unsupported"
        print(
            f"{result.case_name}: predicted={predicted} expected={expected} "
            f"correct={result.correct} provenance={result.provenance_observation_ids} "
            f"superseded={result.superseded_observation_ids}"
        )


if __name__ == "__main__":
    run_demo()
