from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageModelingExample:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")


def language_modeling_corpus_from_examples(
    examples: list[LanguageModelingExample] | tuple[LanguageModelingExample, ...],
) -> str:
    if not examples:
        raise ValueError("examples must not be empty")
    return "\n".join(example.text for example in examples)
