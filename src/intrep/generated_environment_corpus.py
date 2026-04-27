from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from intrep.dataset import ActionConditionedExample
from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl
from intrep.transition_data import generated_find_examples, split_generated_examples
from intrep.types import Action, Fact


EVAL_SLICES = (
    "generated_seen",
    "generated_held_out_object",
    "generated_held_out_container",
    "generated_held_out_location",
)


@dataclass(frozen=True)
class GeneratedEnvironmentCorpusSelection:
    train_documents: list[MixedDocument]
    eval_documents: list[MixedDocument]
    eval_label: str


def generated_environment_pair_documents() -> list[MixedDocument]:
    return environment_pair_documents_from_examples(generated_find_examples())


def generated_environment_corpus_selection(
    eval_slice: str = "generated_held_out_object",
) -> GeneratedEnvironmentCorpusSelection:
    train_documents, eval_documents = generated_environment_train_eval_documents(eval_slice)
    return GeneratedEnvironmentCorpusSelection(
        train_documents=train_documents,
        eval_documents=eval_documents,
        eval_label=eval_slice,
    )


def generated_environment_train_eval_documents(
    eval_slice: str,
) -> tuple[list[MixedDocument], list[MixedDocument]]:
    train_examples, slices = split_generated_examples(generated_find_examples())
    if eval_slice not in slices:
        raise ValueError(f"eval_slice must be one of: {', '.join(EVAL_SLICES)}")
    return (
        environment_pair_documents_from_examples(train_examples),
        environment_pair_documents_from_examples(slices[eval_slice]),
    )


def environment_pair_documents_from_examples(
    examples: Sequence[ActionConditionedExample],
) -> list[MixedDocument]:
    documents: list[MixedDocument] = []
    for example in examples:
        if example.expected_observation is None:
            continue
        documents.extend(
            [
                MixedDocument(
                    id=f"env_pair_symbolic_{example.id}",
                    modality="environment_symbolic",
                    content=_render_symbolic_example(example),
                ),
                MixedDocument(
                    id=f"env_pair_natural_{example.id}",
                    modality="environment_natural",
                    content=_render_natural_example(example),
                ),
            ]
        )
    return documents


def write_generated_environment_pair_jsonl(
    output_path: str | Path,
    documents: Sequence[MixedDocument] | None = None,
) -> list[MixedDocument]:
    output_documents = (
        list(documents)
        if documents is not None
        else generated_environment_pair_documents()
    )
    write_mixed_documents_jsonl(output_path, output_documents)
    return output_documents


def _render_symbolic_example(example: ActionConditionedExample) -> str:
    assert example.expected_observation is not None
    facts = " ; ".join(_render_fact_text(fact) for fact in example.state_before)
    return (
        f"<obs> {facts} "
        f"<action> {_render_action_text(example.action)} "
        f"<next_obs> {_render_fact_text(example.expected_observation)}"
    )


def _render_natural_example(example: ActionConditionedExample) -> str:
    assert example.expected_observation is not None
    object_container = _container_for_object(example)
    location = example.expected_observation.object
    return (
        f"The {example.action.object} is in the {object_container}. "
        f"The {object_container} is at the {location}. "
        f"When {example.action.actor} tries to find the {example.action.object}, "
        f"the next observation places the {example.action.object} at the {location}."
    )


def _container_for_object(example: ActionConditionedExample) -> str:
    for fact in example.state_before:
        if fact.subject == example.action.object and fact.predicate == "located_at":
            return fact.object
    raise ValueError(f"missing object container fact for {example.id}")


def _render_fact_text(fact: Fact) -> str:
    if fact.predicate == "located_at":
        return f"{fact.subject} at {fact.object}"
    return f"{fact.predicate} {fact.subject} {fact.object}"


def _render_action_text(action: Action) -> str:
    if action.type == "find":
        return f"{action.actor} find {action.object}"
    return f"{action.actor} {action.type} {action.object} {action.target}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired symbolic/natural environment MixedDocument JSONL."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument(
        "--eval-slice",
        choices=EVAL_SLICES,
        default="generated_held_out_object",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is None and (args.train_output is None or args.eval_output is None):
        parser.error("provide --output or both --train-output and --eval-output")
    if (args.train_output is None) != (args.eval_output is None):
        parser.error("--train-output and --eval-output must be used together")

    try:
        if args.output is not None:
            documents = write_generated_environment_pair_jsonl(args.output)
            print(f"wrote documents={len(documents)} pairs={len(documents) // 2} to {args.output}")
        if args.train_output is not None and args.eval_output is not None:
            train_documents, eval_documents = generated_environment_train_eval_documents(args.eval_slice)
            write_generated_environment_pair_jsonl(args.train_output, train_documents)
            write_generated_environment_pair_jsonl(args.eval_output, eval_documents)
            print(
                f"wrote train={len(train_documents)} eval={len(eval_documents)} "
                f"eval_slice={args.eval_slice}"
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
