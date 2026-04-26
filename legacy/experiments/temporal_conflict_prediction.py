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


TemporalConflictState = Literal["unsupported", "resolved", "conflict"]


@dataclass(frozen=True)
class TemporalConflictCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]


@dataclass(frozen=True)
class TemporalConflictPrediction:
    state: TemporalConflictState
    fact: Fact | None
    candidates: list[Fact]
    provenance_observation_ids: list[str]
    superseded_observation_ids: list[str]
    conflict_observation_ids: list[str]


@dataclass(frozen=True)
class TemporalConflictResult:
    case_name: str
    prediction_state: TemporalConflictState
    predicted_fact: Fact | None
    candidate_facts: list[Fact]
    expected_candidates: list[Fact]
    correct: bool
    provenance_observation_ids: list[str]
    superseded_observation_ids: list[str]
    conflict_observation_ids: list[str]


@dataclass(frozen=True)
class TemporalConflictSummary:
    results: list[TemporalConflictResult]
    accuracy: float
    conflict_rate: float


class TemporalConflictPredictor:
    def __init__(self, context: list[Observation] | None = None) -> None:
        self.context = list(context or [])

    def predict(self, action: Action) -> TemporalConflictPrediction:
        if action.type != "find":
            return TemporalConflictPrediction(
                state="unsupported",
                fact=None,
                candidates=[],
                provenance_observation_ids=[],
                superseded_observation_ids=[],
                conflict_observation_ids=[],
            )

        current = action.object
        seen = {current}
        provenance: list[str] = []
        superseded: list[str] = []
        destination = None

        while True:
            latest, replaced = self._latest_location_observations(current)
            superseded.extend(observation.id for observation in replaced)
            if not latest:
                return _resolved_or_unsupported(action.object, destination, provenance, superseded)

            facts = [_parse_rendered_fact(observation.content) for observation in latest]
            facts = [fact for fact in facts if fact is not None]
            objects = sorted({fact.object for fact in facts})

            if len(objects) > 1:
                candidates = [Fact(subject=action.object, predicate="located_at", object=object_) for object_ in objects]
                return TemporalConflictPrediction(
                    state="conflict",
                    fact=None,
                    candidates=candidates,
                    provenance_observation_ids=provenance,
                    superseded_observation_ids=_unique(superseded),
                    conflict_observation_ids=[observation.id for observation in latest],
                )

            selected = latest[-1]
            fact = facts[-1]
            provenance.append(selected.id)
            destination = fact.object
            if destination in seen:
                return _resolved_or_unsupported(action.object, destination, provenance, superseded)
            seen.add(destination)
            current = destination

    def _latest_location_observations(self, subject: str) -> tuple[list[Observation], list[Observation]]:
        matching = []
        for observation in self.context:
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                matching.append(observation)

        if not matching:
            return [], []

        latest_timestamp = max(observation.timestamp for observation in matching)
        latest = [observation for observation in matching if observation.timestamp == latest_timestamp]
        replaced = [observation for observation in matching if observation.timestamp != latest_timestamp]
        return latest, replaced


def evaluate_temporal_conflict_cases(cases: list[TemporalConflictCase]) -> TemporalConflictSummary:
    results = []
    for case in cases:
        memory = _build_memory(case.memory_inputs)
        context = _build_full_location_context(memory)
        prediction = TemporalConflictPredictor(context).predict(case.example.action)
        expected_candidates = case.example.expected_state_after
        results.append(
            TemporalConflictResult(
                case_name=case.example.id,
                prediction_state=prediction.state,
                predicted_fact=prediction.fact,
                candidate_facts=prediction.candidates,
                expected_candidates=expected_candidates,
                correct=_same_fact_set(prediction.candidates, expected_candidates),
                provenance_observation_ids=prediction.provenance_observation_ids,
                superseded_observation_ids=prediction.superseded_observation_ids,
                conflict_observation_ids=prediction.conflict_observation_ids,
            )
        )

    if not results:
        return TemporalConflictSummary(results=[], accuracy=0.0, conflict_rate=0.0)

    return TemporalConflictSummary(
        results=results,
        accuracy=sum(1 for result in results if result.correct) / len(results),
        conflict_rate=sum(1 for result in results if result.prediction_state == "conflict") / len(results),
    )


def smoke_cases() -> list[TemporalConflictCase]:
    return [
        TemporalConflictCase(
            example=ActionConditionedExample(
                id="find_key_with_same_time_container_conflict",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=None,
                expected_state_after=[
                    Fact(subject="鍵", predicate="located_at", object="机"),
                    Fact(subject="鍵", predicate="located_at", object="棚"),
                ],
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"], timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"], timestamp="t3"),
                MemoryInput(content="located_at(箱, 机)", tags=["箱", "location"], timestamp="t3"),
            ],
        )
    ]


def _resolved_or_unsupported(
    subject: str,
    destination: str | None,
    provenance: list[str],
    superseded: list[str],
) -> TemporalConflictPrediction:
    if destination is None:
        return TemporalConflictPrediction(
            state="unsupported",
            fact=None,
            candidates=[],
            provenance_observation_ids=provenance,
            superseded_observation_ids=_unique(superseded),
            conflict_observation_ids=[],
        )

    fact = Fact(subject=subject, predicate="located_at", object=destination)
    return TemporalConflictPrediction(
        state="resolved",
        fact=fact,
        candidates=[fact],
        provenance_observation_ids=provenance,
        superseded_observation_ids=_unique(superseded),
        conflict_observation_ids=[],
    )


def _build_full_location_context(memory: ObservationMemory) -> list[Observation]:
    return [observation for observation in memory.observations if _parse_rendered_fact(observation.content)]


def _same_fact_set(left: list[Fact], right: list[Fact]) -> bool:
    return {fact.key() for fact in left} == {fact.key() for fact in right}


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def run_demo() -> None:
    summary = evaluate_temporal_conflict_cases(smoke_cases())
    print(f"accuracy={summary.accuracy:.2f} conflict_rate={summary.conflict_rate:.2f}")
    for result in summary.results:
        candidates = ", ".join(fact.render() for fact in result.candidate_facts) or "none"
        expected = ", ".join(fact.render() for fact in result.expected_candidates) or "none"
        print(
            f"{result.case_name}: state={result.prediction_state} "
            f"candidates=[{candidates}] expected=[{expected}] "
            f"correct={result.correct} conflicts={result.conflict_observation_ids}"
        )


if __name__ == "__main__":
    run_demo()
