from __future__ import annotations

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


TextTokenizerKind = Literal["byte", "byte-pair"]


class TextTokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


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
    raise ValueError("tokenizer must be one of: byte, byte-pair")


def text_tokenizer_to_payload(tokenizer: TextTokenizer) -> dict[str, object]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"kind": "byte"}
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
    if kind == "byte-pair":
        tokenizer_json = payload.get("tokenizer_json")
        if not isinstance(tokenizer_json, str):
            raise ValueError("byte-pair tokenizer payload requires tokenizer_json")
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
