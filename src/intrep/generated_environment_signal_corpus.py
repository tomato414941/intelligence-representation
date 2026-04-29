from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from intrep.dataset import ActionConditionedExample
from intrep.generated_environment_corpus import EVAL_SLICES
from intrep.transition_data import (
    generated_find_examples,
    split_generated_examples,
    split_strict_generated_examples,
    strict_generated_examples,
)
from intrep.signal_io import write_signals_jsonl_v2
from intrep.signals import Signal
from intrep.types import Action, Fact


HARD_NEGATIVE_EVAL_SLICES = (
    "same_history_different_action",
    "same_action_different_context",
)
SIGNAL_EVAL_SLICES = (*EVAL_SLICES, *HARD_NEGATIVE_EVAL_SLICES)


@dataclass(frozen=True)
class GeneratedEnvironmentSignalCorpusSelection:
    train_events: list[Signal]
    eval_events: list[Signal]
    eval_label: str


def generated_environment_signal_corpus_selection(
    eval_slice: str = "generated_held_out_object",
    *,
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> GeneratedEnvironmentSignalCorpusSelection:
    train_events, eval_events = generated_environment_train_eval_events(
        eval_slice,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    return GeneratedEnvironmentSignalCorpusSelection(
        train_events=train_events,
        eval_events=eval_events,
        eval_label=eval_slice,
    )


def write_generated_environment_signal_jsonl(
    train_output: str | Path,
    eval_output: str | Path,
    *,
    eval_slice: str = "generated_held_out_object",
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> GeneratedEnvironmentSignalCorpusSelection:
    selection = generated_environment_signal_corpus_selection(
        eval_slice,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    write_signals_jsonl_v2(train_output, selection.train_events)
    write_signals_jsonl_v2(eval_output, selection.eval_events)
    return selection


def generated_environment_train_eval_events(
    eval_slice: str,
    *,
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> tuple[list[Signal], list[Signal]]:
    if eval_slice in HARD_NEGATIVE_EVAL_SLICES:
        return generated_hard_negative_train_eval_events(
            eval_slice,
            train_size=train_size,
            eval_size=eval_size,
            seed=seed,
        )
    if eval_slice.startswith("generated_strict_"):
        train_examples, slices = split_strict_generated_examples(strict_generated_examples())
    else:
        train_examples, slices = split_generated_examples(generated_find_examples())
    if eval_slice not in slices:
        raise ValueError(f"eval_slice must be one of: {', '.join(SIGNAL_EVAL_SLICES)}")
    return (
        environment_signals_from_examples(
            train_examples,
            split="train",
            eval_slice=eval_slice,
        ),
        environment_signals_from_examples(
            slices[eval_slice],
            split="eval",
            eval_slice=eval_slice,
        ),
    )


def generated_hard_negative_train_eval_events(
    split_name: str,
    *,
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> tuple[list[Signal], list[Signal]]:
    if split_name == "same_history_different_action":
        builder = _same_history_different_action_pair
    elif split_name == "same_action_different_context":
        builder = _same_action_different_context_pair
    else:
        raise ValueError(f"split_name must be one of: {', '.join(HARD_NEGATIVE_EVAL_SLICES)}")
    if train_size < 2 or eval_size < 2:
        raise ValueError("train_size and eval_size must each be at least 2")

    train_pair_count = _required_pair_count(train_size)
    eval_pair_count = _required_pair_count(eval_size)
    pair_specs = _hard_negative_pair_specs(train_pair_count + eval_pair_count, seed=seed)
    return (
        _events_from_hard_negative_specs(
            pair_specs[:train_pair_count],
            split="train",
            split_name=split_name,
            builder=builder,
        ),
        _events_from_hard_negative_specs(
            pair_specs[train_pair_count:],
            split="eval",
            split_name=split_name,
            builder=builder,
        ),
    )


def _required_pair_count(size: int) -> int:
    return max(1, (size + 1) // 2)


def _hard_negative_pair_specs(count: int, *, seed: int) -> list[tuple[str, str, str, str]]:
    objects = ["key", "coin", "map", "watch", "book", "wallet", "ring", "card"]
    containers = ["box_a", "box_b", "drawer_a", "drawer_b", "bag_a", "bag_b"]
    specs: list[tuple[str, str, str, str]] = []
    for left_index, left_object in enumerate(objects):
        for right_index, right_object in enumerate(objects):
            if left_object == right_object:
                continue
            left_container = containers[left_index % len(containers)]
            right_container = containers[(left_index + right_index + 1) % len(containers)]
            if left_container == right_container:
                next_index = (containers.index(right_container) + 1) % len(containers)
                right_container = containers[next_index]
            specs.append((left_object, right_object, left_container, right_container))
    if count > len(specs):
        specs.extend(_synthetic_hard_negative_pair_specs(count - len(specs)))
    random.Random(seed).shuffle(specs)
    return specs[:count]


def _synthetic_hard_negative_pair_specs(count: int) -> list[tuple[str, str, str, str]]:
    return [
        (
            f"object_{index:04d}_left",
            f"object_{index:04d}_right",
            f"box_{index:04d}_left",
            f"box_{index:04d}_right",
        )
        for index in range(count)
    ]


def _events_from_hard_negative_specs(
    specs: Sequence[tuple[str, str, str, str]],
    *,
    split: str,
    split_name: str,
    builder: Callable[..., tuple[list[Signal], list[Signal]]],
) -> list[Signal]:
    events: list[Signal] = []
    for index, (left_object, right_object, left_container, right_container) in enumerate(specs):
        pair_id = f"{split}_{split_name}_{index:04d}"
        left_events, right_events = builder(
            pair_id=pair_id,
            split=split,
            split_name=split_name,
            left_object=left_object,
            right_object=right_object,
            left_container=left_container,
            right_container=right_container,
        )
        events.extend(left_events)
        events.extend(right_events)
    return events


def _same_history_different_action_pair(
    *,
    pair_id: str,
    split: str,
    split_name: str,
    left_object: str,
    right_object: str,
    left_container: str,
    right_container: str,
) -> tuple[list[Signal], list[Signal]]:
    observation = f"{left_container} contains {left_object} ; {right_container} contains {right_object}"
    left_episode = f"{pair_id}_open_left"
    right_episode = f"{pair_id}_open_right"
    return (
        _hard_negative_episode(
            episode_id=left_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=observation,
            action=f"open {left_container}",
            consequence=f"see {left_object}",
        ),
        _hard_negative_episode(
            episode_id=right_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=observation,
            action=f"open {right_container}",
            consequence=f"see {right_object}",
        ),
    )


def _same_action_different_context_pair(
    *,
    pair_id: str,
    split: str,
    split_name: str,
    left_object: str,
    right_object: str,
    left_container: str,
    right_container: str,
) -> tuple[list[Signal], list[Signal]]:
    del left_container, right_container
    left_episode = f"{pair_id}_left_context"
    right_episode = f"{pair_id}_right_context"
    return (
        _hard_negative_episode(
            episode_id=left_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=f"box contains {left_object}",
            action="open box",
            consequence=f"see {left_object}",
        ),
        _hard_negative_episode(
            episode_id=right_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=f"box contains {right_object}",
            action="open box",
            consequence=f"see {right_object}",
        ),
    )


def _hard_negative_episode(
    *,
    episode_id: str,
    split: str,
    split_name: str,
    pair_id: str,
    observation: str,
    action: str,
    consequence: str,
) -> list[Signal]:
    del episode_id, split, split_name, pair_id
    return [
        Signal(
            channel="observation",
            payload=observation,
        ),
        Signal(
            channel="action",
            payload=action,
        ),
        Signal(
            channel="consequence",
            payload=consequence,
        ),
    ]


def environment_signals_from_examples(
    examples: Sequence[ActionConditionedExample],
    *,
    split: str,
    eval_slice: str,
) -> list[Signal]:
    events: list[Signal] = []
    for example in examples:
        consequence = _render_consequence(example)
        events.extend(
            [
                _signal(
                    example,
                    channel="observation",
                    payload=_render_observation(example),
                    split=split,
                    eval_slice=eval_slice,
                ),
                _signal(
                    example,
                    channel="action",
                    payload=_render_action_text(example.action),
                    split=split,
                    eval_slice=eval_slice,
                ),
                _signal(
                    example,
                    channel="consequence",
                    payload=consequence,
                    split=split,
                    eval_slice=eval_slice,
                ),
            ]
        )
    return events


def _signal(
    example: ActionConditionedExample,
    *,
    channel: str,
    payload: str,
    split: str,
    eval_slice: str,
) -> Signal:
    del example, split, eval_slice
    return Signal(
        channel=channel,
        payload=payload,
    )


def _render_observation(example: ActionConditionedExample) -> str:
    return " ; ".join(_render_fact_text(fact) for fact in example.state_before)


def _render_consequence(example: ActionConditionedExample) -> str:
    if example.expected_observation is None:
        return "none"
    return _render_fact_text(example.expected_observation)


def _render_fact_text(fact: Fact) -> str:
    if fact.predicate == "located_at":
        return f"{fact.subject} at {fact.object}"
    return f"{fact.predicate} {fact.subject} {fact.object}"


def _render_action_text(action: Action) -> str:
    if action.type == "find":
        return f"{action.actor} find {action.object}"
    return f"{action.actor} {action.type} {action.object} {action.target}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal observation/action/consequence environment JSONL."
    )
    parser.add_argument("--train-output", required=True, help="Path for train Signal JSONL.")
    parser.add_argument("--eval-output", required=True, help="Path for eval Signal JSONL.")
    parser.add_argument(
        "--eval-slice",
        default="generated_held_out_object",
        choices=SIGNAL_EVAL_SLICES,
        help="Generated environment slice to reserve for eval.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=8,
        help="Approximate number of train episodes for hard-negative signal slices.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=4,
        help="Approximate number of eval episodes for hard-negative signal slices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for hard-negative signal slices.",
    )
    args = parser.parse_args(argv)

    selection = write_generated_environment_signal_jsonl(
        args.train_output,
        args.eval_output,
        eval_slice=args.eval_slice,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    print("intrep generated-environment signal corpus")
    print(f"eval_slice={selection.eval_label}")
    print(f"train_events={len(selection.train_events)}")
    print(f"eval_events={len(selection.eval_events)}")
    print(f"train_output={args.train_output}")
    print(f"eval_output={args.eval_output}")


if __name__ == "__main__":
    main()
