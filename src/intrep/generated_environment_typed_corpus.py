from __future__ import annotations

import argparse
from collections.abc import Sequence
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


@dataclass(frozen=True)
class GeneratedEnvironmentTypedCorpusSelection:
    train_events: list[TypedEvent]
    eval_events: list[TypedEvent]
    eval_label: str


def generated_environment_typed_corpus_selection(
    eval_slice: str = "generated_held_out_object",
) -> GeneratedEnvironmentTypedCorpusSelection:
    train_events, eval_events = generated_environment_train_eval_events(eval_slice)
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
) -> GeneratedEnvironmentTypedCorpusSelection:
    selection = generated_environment_typed_corpus_selection(eval_slice)
    write_typed_events_jsonl_v2(train_output, selection.train_events)
    write_typed_events_jsonl_v2(eval_output, selection.eval_events)
    return selection


def generated_environment_train_eval_events(
    eval_slice: str,
) -> tuple[list[TypedEvent], list[TypedEvent]]:
    if eval_slice.startswith("generated_strict_"):
        train_examples, slices = split_strict_generated_examples(strict_generated_examples())
    else:
        train_examples, slices = split_generated_examples(generated_find_examples())
    if eval_slice not in slices:
        raise ValueError(f"eval_slice must be one of: {', '.join(EVAL_SLICES)}")
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
        choices=EVAL_SLICES,
        help="Generated environment slice to reserve for eval.",
    )
    args = parser.parse_args(argv)

    selection = write_generated_environment_typed_jsonl(
        args.train_output,
        args.eval_output,
        eval_slice=args.eval_slice,
    )
    print("intrep generated-environment typed corpus")
    print(f"eval_slice={selection.eval_label}")
    print(f"train_events={len(selection.train_events)}")
    print(f"eval_events={len(selection.eval_events)}")
    print(f"train_output={args.train_output}")
    print(f"eval_output={args.eval_output}")


if __name__ == "__main__":
    main()
