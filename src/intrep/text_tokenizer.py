from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Literal, Protocol

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from intrep.byte_tokenizer import ByteTokenizer


TextTokenizerKind = Literal["byte", "byte-pair", "simple-byte-pair"]


class TextTokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


@dataclass(frozen=True)
class SimpleBytePairTokenizer:
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


@dataclass(frozen=True)
class BytePairTokenizer:
    tokenizer_json: str

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        return Tokenizer.from_str(self.tokenizer_json)


def build_text_tokenizer(
    text: str,
    *,
    kind: TextTokenizerKind = "byte-pair",
    vocab_size: int = 512,
    min_pair_count: int = 2,
) -> TextTokenizer:
    if kind == "byte":
        return ByteTokenizer()
    if kind == "byte-pair":
        return train_byte_pair_tokenizer(text, vocab_size=vocab_size)
    if kind == "simple-byte-pair":
        return train_simple_byte_pair_tokenizer(
            text,
            vocab_size=vocab_size,
            min_pair_count=min_pair_count,
        )
    raise ValueError("tokenizer must be one of: byte, byte-pair, simple-byte-pair")


def text_tokenizer_to_payload(tokenizer: TextTokenizer) -> dict[str, object]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"kind": "byte"}
    if isinstance(tokenizer, SimpleBytePairTokenizer):
        return {
            "kind": "simple-byte-pair",
            "vocab_size": tokenizer.vocab_size,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    if isinstance(tokenizer, BytePairTokenizer):
        return {
            "kind": "byte-pair",
            "tokenizer_json": tokenizer.tokenizer_json,
            "vocab_size": tokenizer.vocab_size,
        }
    raise TypeError(f"unsupported tokenizer type: {type(tokenizer).__name__}")


def text_tokenizer_from_payload(payload: dict[str, object] | None) -> TextTokenizer:
    if payload is None:
        return ByteTokenizer()
    kind = payload.get("kind")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "simple-byte-pair":
        merges = payload.get("merges")
        if not isinstance(merges, list):
            raise ValueError("simple-byte-pair tokenizer payload requires merges")
        return SimpleBytePairTokenizer(
            merges=tuple(_merge_pair_from_payload(pair) for pair in merges),
            configured_vocab_size=int(payload["vocab_size"]),
        )
    if kind == "byte-pair":
        if "merges" in payload:
            merges = payload.get("merges")
            if not isinstance(merges, list):
                raise ValueError("legacy byte-pair tokenizer payload requires merges")
            return SimpleBytePairTokenizer(
                merges=tuple(_merge_pair_from_payload(pair) for pair in merges),
                configured_vocab_size=int(payload["vocab_size"]),
            )
        tokenizer_json = payload.get("tokenizer_json")
        if not isinstance(tokenizer_json, str):
            raise ValueError("byte-pair tokenizer payload requires tokenizer_json")
        return BytePairTokenizer(tokenizer_json=tokenizer_json)
    if kind == "hf-byte-pair":
        tokenizer_json = payload.get("tokenizer_json")
        if not isinstance(tokenizer_json, str):
            raise ValueError("hf-byte-pair tokenizer payload requires tokenizer_json")
        return BytePairTokenizer(tokenizer_json=tokenizer_json)
    raise ValueError("unsupported tokenizer kind")


def save_text_tokenizer(path: Path, tokenizer: TextTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "intrep.text_tokenizer.v1",
        "tokenizer": text_tokenizer_to_payload(tokenizer),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_text_tokenizer(path: Path) -> TextTokenizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "intrep.text_tokenizer.v1":
        raise ValueError("unsupported text tokenizer schema version")
    tokenizer_payload = payload.get("tokenizer")
    if not isinstance(tokenizer_payload, dict):
        raise ValueError("text tokenizer payload requires tokenizer")
    return text_tokenizer_from_payload(tokenizer_payload)


def train_simple_byte_pair_tokenizer(
    text: str,
    *,
    vocab_size: int = 512,
    min_pair_count: int = 2,
) -> SimpleBytePairTokenizer:
    if vocab_size <= SimpleBytePairTokenizer.pad_id + 1:
        raise ValueError("byte-pair vocab_size must be greater than 257")
    if min_pair_count <= 0:
        raise ValueError("byte-pair min_pair_count must be positive")

    token_ids = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    max_merges = vocab_size - SimpleBytePairTokenizer.pad_id - 1
    for merge_index in range(max_merges):
        pair_counts = Counter(zip(token_ids, token_ids[1:], strict=False))
        if not pair_counts:
            break
        pair, count = pair_counts.most_common(1)[0]
        if count < min_pair_count:
            break
        merged_id = SimpleBytePairTokenizer.pad_id + 1 + merge_index
        token_ids = _apply_merge(token_ids, pair, merged_id)
        merges.append(pair)
    return SimpleBytePairTokenizer(merges=tuple(merges), configured_vocab_size=vocab_size)


def train_byte_pair_tokenizer(
    text: str,
    *,
    vocab_size: int = 512,
) -> BytePairTokenizer:
    if vocab_size <= len(ByteLevel.alphabet()) + 3:
        raise ValueError("byte-pair vocab_size must include the byte alphabet and special tokens")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return BytePairTokenizer(tokenizer_json=tokenizer.to_str())


def _merge_pair_from_payload(pair: object) -> tuple[int, int]:
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(value, int) for value in pair)
    ):
        raise ValueError("byte-pair tokenizer merge must be a pair of integers")
    return (pair[0], pair[1])


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
