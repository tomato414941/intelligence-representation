from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from intrep.byte_tokenizer import ByteTokenizer


TextTokenizerKind = Literal["byte", "byte-pair"]


@dataclass(frozen=True)
class BytePairTokenizer:
    merges: tuple[tuple[int, int], ...] = ()
    configured_vocab_size: int | None = None

    pad_id = 256

    @property
    def vocab_size(self) -> int:
        if self.configured_vocab_size is not None:
            return self.configured_vocab_size
        return self.pad_id + 1 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            merged_id = self.pad_id + 1 + merge_index
            token_ids = _apply_merge(token_ids, pair, merged_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        expanded: list[int] = []
        for token_id in token_ids:
            expanded.extend(self._expand_token(token_id))
        return bytes(expanded).decode("utf-8", errors="replace")

    def _expand_token(self, token_id: int) -> tuple[int, ...]:
        if token_id == self.pad_id:
            return ()
        if 0 <= token_id < 256:
            return (token_id,)
        merge_index = token_id - self.pad_id - 1
        if not 0 <= merge_index < len(self.merges):
            raise ValueError(f"invalid byte-pair token id: {token_id}")
        left, right = self.merges[merge_index]
        return (*self._expand_token(left), *self._expand_token(right))


def build_text_tokenizer(
    text: str,
    *,
    kind: TextTokenizerKind = "byte",
    vocab_size: int = 512,
    min_pair_count: int = 2,
) -> ByteTokenizer | BytePairTokenizer:
    if kind == "byte":
        return ByteTokenizer()
    if kind == "byte-pair":
        return train_byte_pair_tokenizer(
            text,
            vocab_size=vocab_size,
            min_pair_count=min_pair_count,
        )
    raise ValueError("tokenizer must be one of: byte, byte-pair")


def train_byte_pair_tokenizer(
    text: str,
    *,
    vocab_size: int = 512,
    min_pair_count: int = 2,
) -> BytePairTokenizer:
    if vocab_size <= BytePairTokenizer.pad_id + 1:
        raise ValueError("byte-pair vocab_size must be greater than 257")
    if min_pair_count <= 0:
        raise ValueError("byte-pair min_pair_count must be positive")

    token_ids = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    max_merges = vocab_size - BytePairTokenizer.pad_id - 1
    for merge_index in range(max_merges):
        pair_counts = Counter(zip(token_ids, token_ids[1:], strict=False))
        if not pair_counts:
            break
        pair, count = pair_counts.most_common(1)[0]
        if count < min_pair_count:
            break
        merged_id = BytePairTokenizer.pad_id + 1 + merge_index
        token_ids = _apply_merge(token_ids, pair, merged_id)
        merges.append(pair)
    return BytePairTokenizer(merges=tuple(merges), configured_vocab_size=vocab_size)


def _apply_merge(token_ids: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(token_ids):
        if (
            index < len(token_ids) - 1
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        ):
            output.append(merged_id)
            index += 2
        else:
            output.append(token_ids[index])
            index += 1
    return output
