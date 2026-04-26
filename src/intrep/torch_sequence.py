from __future__ import annotations

from dataclasses import dataclass

from intrep.sequence import SequenceExample


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class SequenceVocabulary:
    token_to_id: dict[str, int]
    target_to_id: dict[str, int]
    id_to_target: dict[int, str]

    def encode_tokens(self, tokens: list[str], max_length: int) -> list[int]:
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens]
        return ids[:max_length] + [self.token_to_id[PAD_TOKEN]] * max(0, max_length - len(ids))

    def encode_target(self, target: str) -> int:
        return self.target_to_id[target]

    def decode_target(self, target_id: int) -> str:
        return self.id_to_target[target_id]


def build_vocabulary(sequences: list[SequenceExample]) -> SequenceVocabulary:
    input_tokens = sorted({token for sequence in sequences for token in sequence.input_tokens})
    targets = sorted({sequence.target_token for sequence in sequences})
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in input_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    target_to_id = {target: index for index, target in enumerate(targets)}
    return SequenceVocabulary(
        token_to_id=token_to_id,
        target_to_id=target_to_id,
        id_to_target={index: target for target, index in target_to_id.items()},
    )


def max_input_length(sequences: list[SequenceExample]) -> int:
    if not sequences:
        return 1
    return max(len(sequence.input_tokens) for sequence in sequences)
