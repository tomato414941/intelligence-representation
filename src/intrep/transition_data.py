from __future__ import annotations

from dataclasses import dataclass

from intrep.dataset import ActionConditionedExample
from intrep.environment import MiniTransitionEnvironment
from intrep.evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
from intrep.predictors import FrequencyTransitionPredictor, RuleBasedPredictor
from intrep.types import Action, Fact


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


def held_out_object_examples() -> list[ActionConditionedExample]:
    cases = [
        ("unseen_wallet_find", "財布", "ケース", "引き出し"),
        ("unseen_watch_find", "時計", "ポーチ", "机"),
        ("unseen_ring_find", "指輪", "小箱", "棚"),
    ]

    examples = []
    for case_id, object_name, container, location in cases:
        examples.append(
            ActionConditionedExample(
                id=case_id,
                state_before=[
                    Fact(subject=container, predicate="located_at", object=location),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                action=Action(type="find", actor="太郎", object=object_name, target="unknown"),
                expected_observation=Fact(subject=object_name, predicate="located_at", object=location),
                expected_state_after=[
                    Fact(subject=container, predicate="located_at", object=location),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                source="held_out_object",
            )
        )
    return examples


def longer_chain_examples() -> list[ActionConditionedExample]:
    cases = [
        ("long_chain_badge_find", "社員証", "封筒", "バッグ", "ロッカー"),
        ("long_chain_ticket_find", "切符", "財布", "上着", "玄関"),
    ]
    examples = []
    for case_id, object_name, container, outer_container, location in cases:
        examples.append(
            ActionConditionedExample(
                id=case_id,
                state_before=[
                    Fact(subject=outer_container, predicate="located_at", object=location),
                    Fact(subject=container, predicate="located_at", object=outer_container),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                action=Action(type="find", actor="太郎", object=object_name, target="unknown"),
                expected_observation=Fact(subject=object_name, predicate="located_at", object=location),
                expected_state_after=[
                    Fact(subject=outer_container, predicate="located_at", object=location),
                    Fact(subject=container, predicate="located_at", object=outer_container),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                source="longer_chain",
            )
        )
    return examples


def missing_link_examples() -> list[ActionConditionedExample]:
    cases = [
        ("missing_link_card_find", "カード", "箱"),
        ("missing_link_keycard_find", "キー カード", "ケース"),
    ]
    examples = []
    for case_id, object_name, container in cases:
        examples.append(
            ActionConditionedExample(
                id=case_id,
                state_before=[
                    Fact(subject=container, predicate="contains", object=object_name),
                ],
                action=Action(type="find", actor="太郎", object=object_name, target="unknown"),
                expected_observation=None,
                expected_state_after=[],
                source="missing_link",
            )
        )
    return examples


def noisy_distractor_examples() -> list[ActionConditionedExample]:
    cases = [
        ("noisy_usb_find", "USB", "袋", "引き出し"),
        ("noisy_note_find", "メモ", "手帳", "鞄"),
    ]
    distractors = [
        Fact(subject="眼鏡", predicate="located_at", object="棚"),
        Fact(subject="充電器", predicate="located_at", object="机"),
        Fact(subject="鍵", predicate="located_at", object="箱"),
    ]
    examples = []
    for case_id, object_name, container, location in cases:
        examples.append(
            ActionConditionedExample(
                id=case_id,
                state_before=[
                    *distractors,
                    Fact(subject=container, predicate="located_at", object=location),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                action=Action(type="find", actor="太郎", object=object_name, target="unknown"),
                expected_observation=Fact(subject=object_name, predicate="located_at", object=location),
                expected_state_after=[
                    *distractors,
                    Fact(subject=container, predicate="located_at", object=location),
                    Fact(subject=object_name, predicate="located_at", object=container),
                ],
                source="noisy_distractor",
            )
        )
    return examples


def generated_find_examples() -> list[ActionConditionedExample]:
    objects = ["鍵", "本", "財布", "時計", "指輪", "カード"]
    containers = ["箱", "袋", "ケース", "ポーチ"]
    locations = ["棚", "机", "引き出し"]
    examples = []
    for object_name in objects:
        for container in containers:
            for location in locations:
                examples.append(_find_example("generated", object_name, container, location))
    return examples


def split_generated_examples(
    examples: list[ActionConditionedExample],
) -> tuple[list[ActionConditionedExample], dict[str, list[ActionConditionedExample]]]:
    train = [
        example
        for example in examples
        if (
            example.action.object in {"鍵", "本", "財布"}
            and _container_from_example(example) in {"箱", "袋"}
            and example.expected_observation is not None
            and example.expected_observation.object in {"棚", "机"}
        )
    ]
    slices = {
        "generated_seen": [
            example
            for example in examples
            if (
                example.action.object in {"鍵", "本", "財布"}
                and _container_from_example(example) in {"箱", "袋"}
                and example.expected_observation is not None
                and example.expected_observation.object in {"棚", "机"}
            )
        ],
        "generated_held_out_object": [
            example
            for example in examples
            if example.action.object in {"時計", "指輪", "カード"}
            and _container_from_example(example) in {"箱", "袋"}
            and example.expected_observation is not None
            and example.expected_observation.object in {"棚", "机"}
        ],
        "generated_held_out_container": [
            example
            for example in examples
            if example.action.object in {"鍵", "本", "財布"}
            and _container_from_example(example) in {"ケース", "ポーチ"}
            and example.expected_observation is not None
            and example.expected_observation.object in {"棚", "机"}
        ],
        "generated_held_out_location": [
            example
            for example in examples
            if example.action.object in {"鍵", "本", "財布"}
            and _container_from_example(example) in {"箱", "袋"}
            and example.expected_observation is not None
            and example.expected_observation.object == "引き出し"
        ],
    }
    return train, slices


def _find_example(prefix: str, object_name: str, container: str, location: str) -> ActionConditionedExample:
    return ActionConditionedExample(
        id=f"{prefix}_{object_name}_{container}_{location}",
        state_before=[
            Fact(subject=container, predicate="located_at", object=location),
            Fact(subject=object_name, predicate="located_at", object=container),
        ],
        action=Action(type="find", actor="太郎", object=object_name, target="unknown"),
        expected_observation=Fact(subject=object_name, predicate="located_at", object=location),
        expected_state_after=[
            Fact(subject=container, predicate="located_at", object=location),
            Fact(subject=object_name, predicate="located_at", object=container),
        ],
        source=prefix,
    )


def _container_from_example(example: ActionConditionedExample) -> str:
    for fact in example.state_before:
        if fact.subject == example.action.object and fact.predicate == "located_at":
            return fact.object
    raise ValueError(f"No container fact for {example.id}.")


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
