from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

try:
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import MemoryInput, _parse_rendered_fact
    from experiments.observation_memory import Observation
    from experiments.predictor_interface import Action, Fact
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.observation_assisted_prediction import MemoryInput, _parse_rendered_fact
    from experiments.observation_memory import Observation
    from experiments.predictor_interface import Action, Fact


ReliabilityState = Literal["unsupported", "resolved", "resolved_with_uncertainty", "conflict"]


@dataclass(frozen=True)
class ReliabilityCase:
    example: ActionConditionedExample
    memory_inputs: list[MemoryInput]
    reliability_by_source: dict[str, float]


@dataclass(frozen=True)
class ReliabilityPrediction:
    state: ReliabilityState
    fact: Fact | None
    confidence: float
    candidates: list[Fact]
    counter_candidates: list[Fact]
    provenance_observation_ids: list[str]
    counterevidence_observation_ids: list[str]
    conflict_observation_ids: list[str]


@dataclass(frozen=True)
class ReliabilityResult:
    case_name: str
    prediction_state: ReliabilityState
    predicted_fact: Fact | None
    expected_fact: Fact | None
    correct: bool
    confidence: float
    counter_candidates: list[Fact]
    provenance_observation_ids: list[str]
    counterevidence_observation_ids: list[str]
    conflict_observation_ids: list[str]


@dataclass(frozen=True)
class ReliabilitySummary:
    results: list[ReliabilityResult]
    accuracy: float
    conflict_rate: float
    resolved_with_uncertainty_rate: float


class ReliabilityWeightedPredictor:
    def __init__(
        self,
        context: list[Observation] | None = None,
        *,
        reliability_by_source: dict[str, float] | None = None,
        resolve_margin: float = 0.25,
    ) -> None:
        self.context = list(context or [])
        self.reliability_by_source = reliability_by_source or {}
        self.resolve_margin = resolve_margin

    def predict(self, action: Action) -> ReliabilityPrediction:
        if action.type != "find":
            return _unsupported()

        current = action.object
        seen = {current}
        provenance: list[str] = []

        while True:
            latest = self._latest_location_observations(current)
            if not latest:
                return _resolved_or_unsupported(action.object, None, 0.0, provenance, [], [])

            selected, counter = self._select_by_reliability(latest)
            if selected is None:
                facts = [_parse_rendered_fact(observation.content) for observation in latest]
                candidates = [
                    Fact(subject=action.object, predicate="located_at", object=fact.object)
                    for fact in facts
                    if fact is not None
                ]
                return ReliabilityPrediction(
                    state="conflict",
                    fact=None,
                    confidence=0.0,
                    candidates=_unique_facts(candidates),
                    counter_candidates=[],
                    provenance_observation_ids=provenance,
                    counterevidence_observation_ids=[],
                    conflict_observation_ids=[observation.id for observation in latest],
                )

            fact = _parse_rendered_fact(selected.content)
            if fact is None:
                return _unsupported()

            provenance.append(selected.id)
            destination = fact.object
            counter_facts = _counter_facts(action.object, counter)
            counter_ids = [observation.id for observation in counter]
            confidence = self._reliability(selected)

            if destination in seen:
                return _resolved_or_unsupported(
                    action.object,
                    destination,
                    confidence,
                    provenance,
                    counter_facts,
                    counter_ids,
                )
            seen.add(destination)

            next_observations = self._latest_location_observations(destination)
            if not next_observations:
                return _resolved_or_unsupported(
                    action.object,
                    destination,
                    confidence,
                    provenance,
                    counter_facts,
                    counter_ids,
                )
            current = destination

    def _latest_location_observations(self, subject: str) -> list[Observation]:
        matching = []
        for observation in self.context:
            fact = _parse_rendered_fact(observation.content)
            if fact and fact.subject == subject and fact.predicate == "located_at":
                matching.append(observation)
        if not matching:
            return []
        latest_timestamp = max(observation.timestamp for observation in matching)
        return [observation for observation in matching if observation.timestamp == latest_timestamp]

    def _select_by_reliability(self, observations: list[Observation]) -> tuple[Observation | None, list[Observation]]:
        if len(observations) == 1:
            return observations[0], []

        ranked = sorted(observations, key=self._reliability, reverse=True)
        best = ranked[0]
        second = ranked[1]
        if self._reliability(best) - self._reliability(second) < self.resolve_margin:
            return None, observations
        return best, ranked[1:]

    def _reliability(self, observation: Observation) -> float:
        return self.reliability_by_source.get(observation.source, 0.5)


def evaluate_reliability_cases(cases: list[ReliabilityCase], *, resolve_margin: float = 0.25) -> ReliabilitySummary:
    results = []
    for case in cases:
        context = _build_context(case.memory_inputs)
        prediction = ReliabilityWeightedPredictor(
            context,
            reliability_by_source=case.reliability_by_source,
            resolve_margin=resolve_margin,
        ).predict(case.example.action)
        expected = case.example.expected_observation
        results.append(
            ReliabilityResult(
                case_name=case.example.id,
                prediction_state=prediction.state,
                predicted_fact=prediction.fact,
                expected_fact=expected,
                correct=prediction.fact is not None and expected is not None and prediction.fact.key() == expected.key(),
                confidence=prediction.confidence,
                counter_candidates=prediction.counter_candidates,
                provenance_observation_ids=prediction.provenance_observation_ids,
                counterevidence_observation_ids=prediction.counterevidence_observation_ids,
                conflict_observation_ids=prediction.conflict_observation_ids,
            )
        )

    if not results:
        return ReliabilitySummary(results=[], accuracy=0.0, conflict_rate=0.0, resolved_with_uncertainty_rate=0.0)

    return ReliabilitySummary(
        results=results,
        accuracy=sum(1 for result in results if result.correct) / len(results),
        conflict_rate=sum(1 for result in results if result.prediction_state == "conflict") / len(results),
        resolved_with_uncertainty_rate=sum(
            1 for result in results if result.prediction_state == "resolved_with_uncertainty"
        )
        / len(results),
    )


def smoke_cases() -> list[ReliabilityCase]:
    return [
        ReliabilityCase(
            example=ActionConditionedExample(
                id="high_reliability_sensor_resolves_conflict",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=Fact(subject="鍵", predicate="located_at", object="棚"),
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"], timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"], timestamp="t3"),
                MemoryInput(content="located_at(箱, 机)", tags=["箱", "location"], timestamp="t3"),
            ],
            reliability_by_source={"fixture_1": 0.9, "fixture_2": 0.95, "fixture_3": 0.4},
        ),
        ReliabilityCase(
            example=ActionConditionedExample(
                id="similar_reliability_stays_conflict",
                state_before=[],
                action=Action(type="find", actor="太郎", object="鍵", target="unknown"),
                expected_observation=None,
            ),
            memory_inputs=[
                MemoryInput(content="located_at(鍵, 箱)", tags=["鍵", "location"], timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", tags=["箱", "location"], timestamp="t3"),
                MemoryInput(content="located_at(箱, 机)", tags=["箱", "location"], timestamp="t3"),
            ],
            reliability_by_source={"fixture_1": 0.9, "fixture_2": 0.75, "fixture_3": 0.7},
        ),
    ]


def _build_context(memory_inputs: list[MemoryInput]) -> list[Observation]:
    observations = []
    for index, memory_input in enumerate(memory_inputs, start=1):
        observation = Observation(
            id=f"obs_{index}",
            content=memory_input.content,
            modality="text",
            source=f"fixture_{index}",
            timestamp=memory_input.timestamp,
            tags=memory_input.tags,
        )
        observations.append(observation)
    return observations


def _resolved_or_unsupported(
    subject: str,
    destination: str | None,
    confidence: float,
    provenance: list[str],
    counter_candidates: list[Fact],
    counterevidence_ids: list[str],
) -> ReliabilityPrediction:
    if destination is None:
        return _unsupported()
    fact = Fact(subject=subject, predicate="located_at", object=destination)
    return ReliabilityPrediction(
        state="resolved_with_uncertainty" if counter_candidates else "resolved",
        fact=fact,
        confidence=confidence,
        candidates=[fact],
        counter_candidates=counter_candidates,
        provenance_observation_ids=provenance,
        counterevidence_observation_ids=counterevidence_ids,
        conflict_observation_ids=[],
    )


def _unsupported() -> ReliabilityPrediction:
    return ReliabilityPrediction(
        state="unsupported",
        fact=None,
        confidence=0.0,
        candidates=[],
        counter_candidates=[],
        provenance_observation_ids=[],
        counterevidence_observation_ids=[],
        conflict_observation_ids=[],
    )


def _counter_facts(subject: str, observations: list[Observation]) -> list[Fact]:
    facts = []
    for observation in observations:
        fact = _parse_rendered_fact(observation.content)
        if fact is not None:
            facts.append(Fact(subject=subject, predicate="located_at", object=fact.object))
    return _unique_facts(facts)


def _unique_facts(facts: list[Fact]) -> list[Fact]:
    by_key = {fact.key(): fact for fact in facts}
    return [by_key[key] for key in sorted(by_key)]


def run_demo() -> None:
    summary = evaluate_reliability_cases(smoke_cases())
    print(
        f"accuracy={summary.accuracy:.2f} conflict_rate={summary.conflict_rate:.2f} "
        f"resolved_with_uncertainty_rate={summary.resolved_with_uncertainty_rate:.2f}"
    )
    for result in summary.results:
        predicted = result.predicted_fact.render() if result.predicted_fact else "none"
        counters = ", ".join(fact.render() for fact in result.counter_candidates) or "none"
        print(
            f"{result.case_name}: state={result.prediction_state} predicted={predicted} "
            f"confidence={result.confidence:.2f} counter=[{counters}] "
            f"correct={result.correct} provenance={result.provenance_observation_ids} "
            f"counterevidence={result.counterevidence_observation_ids} conflicts={result.conflict_observation_ids}"
        )


if __name__ == "__main__":
    run_demo()
