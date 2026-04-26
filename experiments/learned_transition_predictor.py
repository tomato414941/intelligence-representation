from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.predictor_evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.action_conditioned_dataset import ActionConditionedExample
    from experiments.predictor_evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor


@dataclass(frozen=True)
class LearnedTransitionComparison:
    train_size: int
    test_size: int
    rule_summary: PredictionEvaluationSummary
    learned_summary: PredictionEvaluationSummary


class MiniTransitionEnvironment:
    def __init__(self, initial_locations: dict[str, str] | None = None) -> None:
        self.locations = dict(initial_locations or {})

    def apply(self, action: Action) -> Fact | None:
        if action.type == "place":
            self.locations[action.object] = action.target
            return Fact(subject=action.object, predicate="located_at", object=action.target)

        if action.type == "move_container":
            self.locations[action.object] = action.target
            return Fact(subject=action.object, predicate="located_at", object=action.target)

        if action.type == "find":
            location = self.resolve_location(action.object)
            if location is None:
                return None
            return Fact(subject=action.object, predicate="located_at", object=location)

        return None

    def resolve_location(self, object_name: str) -> str | None:
        current = object_name
        seen = {current}
        destination = self.locations.get(current)

        while destination in self.locations and destination not in seen:
            seen.add(destination)
            current = destination
            destination = self.locations.get(current)

        return destination

    def facts(self) -> list[Fact]:
        return [
            Fact(subject=subject, predicate="located_at", object=location)
            for subject, location in sorted(self.locations.items())
        ]


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


def generate_examples() -> list[ActionConditionedExample]:
    actions = [
        Action(type="find", actor="太郎", object="鍵", target="unknown"),
        Action(type="move_container", actor="太郎", object="箱", target="机"),
        Action(type="find", actor="太郎", object="鍵", target="unknown"),
        Action(type="place", actor="太郎", object="本", target="図書館"),
        Action(type="find", actor="太郎", object="本", target="unknown"),
        Action(type="move_container", actor="太郎", object="箱", target="棚"),
    ]

    examples = []
    next_id = 1
    for _ in range(2):
        environment = MiniTransitionEnvironment({"鍵": "箱", "箱": "棚"})
        for action in actions:
            state_before = environment.facts()
            observation = environment.apply(action)
            examples.append(
                ActionConditionedExample(
                    id=f"env_case_{next_id}",
                    state_before=state_before,
                    action=action,
                    expected_observation=observation,
                    expected_state_after=environment.facts(),
                    source="mini_transition_environment",
                )
            )
            next_id += 1
    return examples


def split_examples(
    examples: list[ActionConditionedExample],
) -> tuple[list[ActionConditionedExample], list[ActionConditionedExample]]:
    train_ids = {"env_case_1", "env_case_2", "env_case_3", "env_case_4", "env_case_5", "env_case_6"}
    train = [example for example in examples if example.id in train_ids]
    test = [example for example in examples if example.id not in train_ids]
    return train, test


def compare_predictors(
    train_examples: list[ActionConditionedExample],
    test_examples: list[ActionConditionedExample],
) -> LearnedTransitionComparison:
    learned = FrequencyTransitionPredictor()
    learned.fit(train_examples)
    test_cases = [example.to_prediction_case() for example in test_examples]

    return LearnedTransitionComparison(
        train_size=len(train_examples),
        test_size=len(test_examples),
        rule_summary=evaluate_prediction_cases(test_cases, RuleBasedPredictor()),
        learned_summary=evaluate_prediction_cases(test_cases, learned),
    )


def smoke_comparison() -> LearnedTransitionComparison:
    train, test = split_examples(generate_examples())
    return compare_predictors(train, test)


def _action_key(action: Action) -> tuple[str, str, str]:
    return (action.type, action.object, action.target)


def _state_key(state: list[Fact]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(fact.key() for fact in state))


def _object_key(action: Action) -> tuple[str, str]:
    return (action.type, action.object)


def _fact_from_count(counts: Counter[tuple[str, str, str]]) -> Fact:
    key, _ = counts.most_common(1)[0]
    return Fact(subject=key[0], predicate=key[1], object=key[2])


def run_demo() -> None:
    comparison = smoke_comparison()
    print(f"train_size={comparison.train_size} test_size={comparison.test_size}")
    print(
        f"rule_accuracy={comparison.rule_summary.accuracy:.2f} "
        f"rule_unsupported={comparison.rule_summary.unsupported_rate:.2f}"
    )
    print(
        f"learned_accuracy={comparison.learned_summary.accuracy:.2f} "
        f"learned_unsupported={comparison.learned_summary.unsupported_rate:.2f}"
    )


if __name__ == "__main__":
    run_demo()
