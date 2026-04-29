from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTextExample:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")


def text_corpus_from_examples(examples: list[RawTextExample] | tuple[RawTextExample, ...]) -> str:
    if not examples:
        raise ValueError("examples must not be empty")
    return "\n".join(example.text for example in examples)
