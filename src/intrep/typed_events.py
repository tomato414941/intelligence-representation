from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class EventRole(StrEnum):
    TEXT = "text"
    OBSERVATION = "observation"
    ACTION = "action"
    CONSEQUENCE = "consequence"
    PREDICTION = "prediction"
    PREDICTION_ERROR = "prediction_error"
    STATE = "state"
    BELIEF = "belief"
    MEMORY = "memory"
    REWARD = "reward"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


EVENT_ROLES = frozenset(role.value for role in EventRole)


@dataclass(frozen=True, init=False)
class TypedEvent:
    role: EventRole
    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        *,
        role: EventRole | str,
        content: str,
        metadata: Mapping[str, object] | None = None,
        id: str | None = None,
        modality: str | None = None,
        time_index: int | None = None,
        episode_id: str | None = None,
        source_id: str | None = None,
    ) -> None:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        merged_metadata = dict(metadata or {})
        for key, value in (
            ("id", id),
            ("modality", modality),
            ("time_index", time_index),
            ("episode_id", episode_id),
            ("source_id", source_id),
        ):
            if value is not None:
                merged_metadata.setdefault(key, value)
        try:
            rendered_role = EventRole(role)
        except ValueError as error:
            roles = ", ".join(sorted(EVENT_ROLES))
            raise ValueError(f"event role must be one of: {roles}") from error
        object.__setattr__(self, "role", rendered_role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", merged_metadata)

    @property
    def id(self) -> str:
        return str(self.metadata.get("id", ""))

    @property
    def modality(self) -> str:
        return str(self.metadata.get("modality", self.metadata.get("type", "")))

    @property
    def time_index(self) -> int | None:
        value = self.metadata.get("time_index")
        return value if isinstance(value, int) else None

    @property
    def episode_id(self) -> str | None:
        value = self.metadata.get("episode_id")
        return str(value) if value is not None else None

    @property
    def source_id(self) -> str | None:
        value = self.metadata.get("source_id")
        return str(value) if value is not None else None


def render_typed_event(event: TypedEvent) -> str:
    attributes = [
        ("id", event.id),
        ("role", event.role.value),
        ("modality", event.modality),
    ]
    if event.episode_id is not None:
        attributes.append(("episode", event.episode_id))
    if event.time_index is not None:
        attributes.append(("t", str(event.time_index)))
    if event.source_id is not None:
        attributes.append(("source", event.source_id))
    rendered_names = {name for name, _ in attributes}
    for key, value in sorted(event.metadata.items()):
        if key in {"id", "modality", "episode_id", "time_index", "source_id", "type"}:
            continue
        attributes.append((_render_attribute_name(key), str(value)))
    rendered_attributes = " ".join(
        f'{name}="{_escape_tag_attribute(value)}"' for name, value in attributes if name not in rendered_names or name in {"id", "role", "modality", "episode", "t", "source"}
    )
    return f"<EVENT {rendered_attributes}>\n{_validate_content(event.content)}</EVENT>\n"


def _render_attribute_name(name: str) -> str:
    if not name or any(not (character.isalnum() or character in "_-:.") for character in name):
        raise ValueError(f"metadata key is not renderable as a tag attribute: {name!r}")
    return name


def _escape_attribute(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_tag_attribute(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError("event tag attributes must not contain whitespace")
    return _escape_attribute(value)


def _validate_content(content: str) -> str:
    if "</EVENT>" in content:
        raise ValueError("event content must not contain </EVENT>")
    return content if content.endswith("\n") else f"{content}\n"
