from __future__ import annotations

from dataclasses import dataclass

from intrep.dataset import ActionConditionedExample
from intrep.tokens import model_input_tokens, target_token


@dataclass(frozen=True)
class SequenceExample:
    id: str
    input_tokens: list[str]
    target_token: str
    source: str


def sequence_from_example(example: ActionConditionedExample) -> SequenceExample:
    return SequenceExample(
        id=example.id,
        input_tokens=model_input_tokens(example.state_before, example.action),
        target_token=target_token(example.to_prediction_case().expected_fact),
        source=example.source,
    )


def sequences_from_examples(examples: list[ActionConditionedExample]) -> list[SequenceExample]:
    return [sequence_from_example(example) for example in examples]
