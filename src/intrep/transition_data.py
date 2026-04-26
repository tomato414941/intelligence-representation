from __future__ import annotations

from dataclasses import dataclass

from intrep.dataset import ActionConditionedExample
from intrep.environment import MiniTransitionEnvironment
from intrep.evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
from intrep.predictors import FrequencyTransitionPredictor, RuleBasedPredictor
from intrep.types import Action


@dataclass(frozen=True)
class LearnedTransitionComparison:
    train_size: int
    test_size: int
    rule_summary: PredictionEvaluationSummary
    learned_summary: PredictionEvaluationSummary


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

