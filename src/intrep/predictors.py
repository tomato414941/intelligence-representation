from __future__ import annotations

from collections import Counter, defaultdict

from intrep.dataset import ActionConditionedExample
from intrep.types import Action, Fact, Predictor


class RuleBasedPredictor:
    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if action.type == "place":
            return Fact(subject=action.object, predicate="located_at", object=action.target)
        return None


class FrequencyTransitionPredictor:
    def __init__(self) -> None:
        self._by_state_action: dict[tuple[tuple[tuple[str, str, str], ...], tuple[str, str, str]], Fact] = {}
        self._by_action: dict[tuple[str, str, str], Fact] = {}
        self._by_type_object: dict[tuple[str, str], Fact] = {}

    def fit(self, examples: list[ActionConditionedExample]) -> None:
        state_action_counts: dict[
            tuple[tuple[tuple[str, str, str], ...], tuple[str, str, str]], Counter[tuple[str, str, str]]
        ] = defaultdict(Counter)
        action_counts: dict[tuple[str, str, str], Counter[tuple[str, str, str]]] = defaultdict(Counter)
        object_counts: dict[tuple[str, str], Counter[tuple[str, str, str]]] = defaultdict(Counter)

        for example in examples:
            expected = example.expected_observation
            if expected is None:
                continue
            state_action_key = (_state_key(example.state_before), _action_key(example.action))
            action_key = _action_key(example.action)
            object_key = _object_key(example.action)
            state_action_counts[state_action_key][expected.key()] += 1
            action_counts[action_key][expected.key()] += 1
            object_counts[object_key][expected.key()] += 1

        self._by_state_action = {key: _fact_from_count(counts) for key, counts in state_action_counts.items()}
        self._by_action = {key: _fact_from_count(counts) for key, counts in action_counts.items()}
        self._by_type_object = {key: _fact_from_count(counts) for key, counts in object_counts.items()}

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        return (
            self._by_state_action.get((_state_key(state), _action_key(action)))
            or self._by_action.get(_action_key(action))
            or self._by_type_object.get(_object_key(action))
        )


class StateAwarePredictor:
    def __init__(self) -> None:
        self.frequency = FrequencyTransitionPredictor()

    def fit(self, examples: list[ActionConditionedExample]) -> None:
        self.frequency.fit(examples)

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if action.type == "find":
            location = _resolve_location(state, action.object)
            if location is not None:
                return Fact(subject=action.object, predicate="located_at", object=location)

        predicted = self.frequency.predict(state, action)
        if predicted is not None:
            return predicted

        if action.type in {"place", "move_container"}:
            return Fact(subject=action.object, predicate="located_at", object=action.target)

        return None


def _action_key(action: Action) -> tuple[str, str, str]:
    return (action.type, action.object, action.target)


def _state_key(state: list[Fact]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(fact.key() for fact in state))


def _object_key(action: Action) -> tuple[str, str]:
    return (action.type, action.object)


def _fact_from_count(counts: Counter[tuple[str, str, str]]) -> Fact:
    key, _ = counts.most_common(1)[0]
    return Fact(subject=key[0], predicate=key[1], object=key[2])


def _resolve_location(state: list[Fact], object_name: str) -> str | None:
    locations = {
        fact.subject: fact.object
        for fact in state
        if fact.predicate == "located_at"
    }
    current = object_name
    seen = {current}
    destination = locations.get(current)
    traversed_edges = 0

    while destination in locations and destination not in seen:
        seen.add(destination)
        current = destination
        destination = locations.get(current)
        traversed_edges += 1

    return destination
