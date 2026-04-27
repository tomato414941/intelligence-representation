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
from intrep.typed_corpus import write_typed_events_jsonl_v2
from intrep.typed_events import EventRole, TypedEvent
from intrep.types import Action, Fact


HARD_NEGATIVE_EVAL_SLICES = (
    "same_history_different_action",
    "same_action_different_context",
)
TYPED_EVAL_SLICES = (*EVAL_SLICES, *HARD_NEGATIVE_EVAL_SLICES)


@dataclass(frozen=True)
class GeneratedEnvironmentTypedCorpusSelection:
    train_events: list[TypedEvent]
    eval_events: list[TypedEvent]
    eval_label: str


def generated_environment_typed_corpus_selection(
    eval_slice: str = "generated_held_out_object",
    *,
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> GeneratedEnvironmentTypedCorpusSelection:
    train_events, eval_events = generated_environment_train_eval_events(
        eval_slice,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    return GeneratedEnvironmentTypedCorpusSelection(
        train_events=train_events,
        eval_events=eval_events,
        eval_label=eval_slice,
    )


def write_generated_environment_typed_jsonl(
    train_output: str | Path,
    eval_output: str | Path,
    *,
    eval_slice: str = "generated_held_out_object",
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> GeneratedEnvironmentTypedCorpusSelection:
    selection = generated_environment_typed_corpus_selection(
        eval_slice,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    write_typed_events_jsonl_v2(train_output, selection.train_events)
    write_typed_events_jsonl_v2(eval_output, selection.eval_events)
    return selection


def generated_environment_train_eval_events(
    eval_slice: str,
    *,
    train_size: int = 8,
    eval_size: int = 4,
    seed: int = 0,
) -> tuple[list[TypedEvent], list[TypedEvent]]:
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
        raise ValueError(f"eval_slice must be one of: {', '.join(TYPED_EVAL_SLICES)}")
    return (
        environment_typed_events_from_examples(
            train_examples,
            split="train",
            eval_slice=eval_slice,
        ),
        environment_typed_events_from_examples(
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
) -> tuple[list[TypedEvent], list[TypedEvent]]:
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
    random.Random(seed).shuffle(specs)
    if count > len(specs):
        raise ValueError(f"requested {count} hard-negative pairs, but only {len(specs)} are available")
    return specs[:count]


def _events_from_hard_negative_specs(
    specs: Sequence[tuple[str, str, str, str]],
    *,
    split: str,
    split_name: str,
    builder: Callable[..., tuple[list[TypedEvent], list[TypedEvent]]],
) -> list[TypedEvent]:
    events: list[TypedEvent] = []
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
) -> tuple[list[TypedEvent], list[TypedEvent]]:
    observation = f"{left_container} contains {left_object} ; {right_container} contains {right_object}"
    left_episode = f"{pair_id}_open_left"
    right_episode = f"{pair_id}_open_right"
    left_consequence_id = _hard_event_id(left_episode, EventRole.CONSEQUENCE)
    right_consequence_id = _hard_event_id(right_episode, EventRole.CONSEQUENCE)
    return (
        _hard_negative_episode(
            episode_id=left_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=observation,
            action=f"open {left_container}",
            consequence=f"see {left_object}",
            negative_event_id=right_consequence_id,
        ),
        _hard_negative_episode(
            episode_id=right_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=observation,
            action=f"open {right_container}",
            consequence=f"see {right_object}",
            negative_event_id=left_consequence_id,
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
) -> tuple[list[TypedEvent], list[TypedEvent]]:
    del left_container, right_container
    left_episode = f"{pair_id}_left_context"
    right_episode = f"{pair_id}_right_context"
    left_consequence_id = _hard_event_id(left_episode, EventRole.CONSEQUENCE)
    right_consequence_id = _hard_event_id(right_episode, EventRole.CONSEQUENCE)
    return (
        _hard_negative_episode(
            episode_id=left_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=f"box contains {left_object}",
            action="open box",
            consequence=f"see {left_object}",
            negative_event_id=right_consequence_id,
        ),
        _hard_negative_episode(
            episode_id=right_episode,
            split=split,
            split_name=split_name,
            pair_id=pair_id,
            observation=f"box contains {right_object}",
            action="open box",
            consequence=f"see {right_object}",
            negative_event_id=left_consequence_id,
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
    negative_event_id: str,
) -> list[TypedEvent]:
    metadata = {
        "split": split,
        "eval_slice": split_name,
        "condition": split_name,
        "pair_id": pair_id,
    }
    return [
        TypedEvent(
            id=_hard_event_id(episode_id, EventRole.OBSERVATION),
            role=EventRole.OBSERVATION,
            modality="environment_symbolic",
            content=observation,
            episode_id=episode_id,
            time_index=0,
            source_id=pair_id,
            metadata=metadata,
        ),
        TypedEvent(
            id=_hard_event_id(episode_id, EventRole.ACTION),
            role=EventRole.ACTION,
            modality="environment_action",
            content=action,
            episode_id=episode_id,
            time_index=1,
            source_id=pair_id,
            metadata=metadata,
        ),
        TypedEvent(
            id=_hard_event_id(episode_id, EventRole.CONSEQUENCE),
            role=EventRole.CONSEQUENCE,
            modality="environment_consequence",
            content=consequence,
            episode_id=episode_id,
            time_index=2,
            source_id=pair_id,
            metadata={**metadata, "negative_event_ids": [negative_event_id]},
        ),
    ]


def _hard_event_id(episode_id: str, role: EventRole) -> str:
    return f"env_typed_{episode_id}_{role.value}"


def environment_typed_events_from_examples(
    examples: Sequence[ActionConditionedExample],
    *,
    split: str,
    eval_slice: str,
) -> list[TypedEvent]:
    consequence_ids_by_content = {
        _render_consequence(example): _event_id(example, EventRole.CONSEQUENCE, split)
        for example in examples
    }
    events: list[TypedEvent] = []
    for example in examples:
        consequence = _render_consequence(example)
        negative_event_ids = sorted(
            event_id
            for content, event_id in consequence_ids_by_content.items()
            if content != consequence
        )
        events.extend(
            [
                _typed_event(
                    example,
                    role=EventRole.OBSERVATION,
                    modality="environment_symbolic",
                    content=_render_observation(example),
                    time_index=0,
                    split=split,
                    eval_slice=eval_slice,
                ),
                _typed_event(
                    example,
                    role=EventRole.ACTION,
                    modality="environment_action",
                    content=_render_action_text(example.action),
                    time_index=1,
                    split=split,
                    eval_slice=eval_slice,
                ),
                _typed_event(
                    example,
                    role=EventRole.CONSEQUENCE,
                    modality="environment_consequence",
                    content=consequence,
                    time_index=2,
                    split=split,
                    eval_slice=eval_slice,
                    metadata={"negative_event_ids": negative_event_ids},
                ),
            ]
        )
    return events


def _typed_event(
    example: ActionConditionedExample,
    *,
    role: EventRole,
    modality: str,
    content: str,
    time_index: int,
    split: str,
    eval_slice: str,
    metadata: dict[str, object] | None = None,
) -> TypedEvent:
    event_metadata: dict[str, object] = {
        "split": split,
        "eval_slice": eval_slice,
        "source": example.source,
        "source_example_id": example.id,
    }
    event_metadata.update(metadata or {})
    return TypedEvent(
        id=_event_id(example, role, split),
        role=role,
        modality=modality,
        content=content,
        episode_id=f"{split}_{example.id}",
        time_index=time_index,
        source_id=example.id,
        metadata=event_metadata,
    )


def _event_id(example: ActionConditionedExample, role: EventRole, split: str) -> str:
    return f"env_typed_{split}_{role.value}_{example.id}"


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
        description="Generate typed observation/action/consequence environment JSONL."
    )
    parser.add_argument("--train-output", required=True, help="Path for train TypedEvent JSONL.")
    parser.add_argument("--eval-output", required=True, help="Path for eval TypedEvent JSONL.")
    parser.add_argument(
        "--eval-slice",
        default="generated_held_out_object",
        choices=TYPED_EVAL_SLICES,
        help="Generated environment slice to reserve for eval.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=8,
        help="Approximate number of train episodes for hard-negative typed slices.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=4,
        help="Approximate number of eval episodes for hard-negative typed slices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for hard-negative typed slices.",
    )
    args = parser.parse_args(argv)

    selection = write_generated_environment_typed_jsonl(
        args.train_output,
        args.eval_output,
        eval_slice=args.eval_slice,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    print("intrep generated-environment typed corpus")
    print(f"eval_slice={selection.eval_label}")
    print(f"train_events={len(selection.train_events)}")
    print(f"eval_events={len(selection.eval_events)}")
    print(f"train_output={args.train_output}")
    print(f"eval_output={args.eval_output}")


if __name__ == "__main__":
    main()
