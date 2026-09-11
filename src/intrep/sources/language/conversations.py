from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant") or not self.content.strip():
            raise ValueError("chat messages require a supported role and nonempty text")


@dataclass(frozen=True)
class ConversationExample:
    """A supervised conversation ending in an assistant response, without motor actions."""

    id: str
    group_id: str
    messages: tuple[ChatMessage, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.id or not self.group_id or not self.source:
            raise ValueError("conversation identity, source group and provenance are required")
        if len(self.messages) < 2 or self.messages[-1].role != "assistant":
            raise ValueError("a training conversation must end in an assistant response")
        if not any(message.role == "user" for message in self.messages[:-1]):
            raise ValueError("a training conversation requires a user message")

    def prompt(self) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in self.messages[:-1]]

    @property
    def answer(self) -> str:
        return self.messages[-1].content


def load_conversations(path: Path) -> list[ConversationExample]:
    examples, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example = ConversationExample(
            id=row["id"], group_id=row["group_id"], source=row["source"],
            messages=tuple(ChatMessage(**message) for message in row["messages"]),
        )
        if example.id in seen:
            raise ValueError("duplicate conversation id")
        seen.add(example.id)
        examples.append(example)
    if not examples:
        raise ValueError("conversation source is empty")
    return examples


def conversation_source(path: Path, examples: list[ConversationExample]) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "conversation_ids": [example.id for example in examples],
            "group_ids": sorted({example.group_id for example in examples})}


def check_conversation_split(train: list[ConversationExample], evaluation: list[ConversationExample]) -> None:
    if ({row.id for row in train}.intersection(row.id for row in evaluation)
            or {row.group_id for row in train}.intersection(row.group_id for row in evaluation)):
        raise ValueError("conversation trees must not cross training and evaluation splits")
