from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    id: str
    role: EventRole
    modality: str
    content: str
    episode_id: str | None = None
    time_index: int | None = None
    source_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        *,
        role: EventRole | str,
        content: str,
        id: str = "",
        modality: str = "",
        episode_id: str | None = None,
        time_index: int | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        auxiliary_metadata = dict(metadata or {})
        metadata_id = auxiliary_metadata.pop("id", "")
        metadata_modality = auxiliary_metadata.pop("modality", "")
        metadata_type = auxiliary_metadata.pop("type", "")
        metadata_episode_id = auxiliary_metadata.pop("episode_id", None)
        metadata_time_index = auxiliary_metadata.pop("time_index", None)
        metadata_source_id = auxiliary_metadata.pop("source_id", None)

        resolved_id = id or str(metadata_id)
        resolved_modality = modality or str(metadata_modality or metadata_type)
        resolved_episode_id = (
            episode_id if episode_id is not None else _optional_string(metadata_episode_id)
        )
        resolved_time_index = time_index
        if resolved_time_index is None and isinstance(metadata_time_index, int):
            resolved_time_index = metadata_time_index
        resolved_source_id = (
            source_id if source_id is not None else _optional_string(metadata_source_id)
        )

        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("event id must be a non-empty string")
        if not isinstance(resolved_modality, str) or not resolved_modality:
            raise ValueError("event modality must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("event content must be a string")
        try:
            rendered_role = EventRole(role)
        except ValueError as error:
            roles = ", ".join(sorted(EVENT_ROLES))
            raise ValueError(f"event role must be one of: {roles}") from error
        if resolved_episode_id is not None and not isinstance(resolved_episode_id, str):
            raise ValueError("event episode_id must be a string or None")
        if resolved_time_index is not None and not isinstance(resolved_time_index, int):
            raise ValueError("event time_index must be an int or None")
        if resolved_source_id is not None and not isinstance(resolved_source_id, str):
            raise ValueError("event source_id must be a string or None")
        object.__setattr__(self, "id", resolved_id)
        object.__setattr__(self, "role", rendered_role)
        object.__setattr__(self, "modality", resolved_modality)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "episode_id", resolved_episode_id)
        object.__setattr__(self, "time_index", resolved_time_index)
        object.__setattr__(self, "source_id", resolved_source_id)
        object.__setattr__(self, "metadata", auxiliary_metadata)


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
    for key, value in sorted(event.metadata.items()):
        attributes.append((_render_attribute_name(key), _render_attribute_value(value)))
    rendered_attributes = " ".join(
        f'{name}="{_escape_tag_attribute(value)}"' for name, value in attributes
    )
    return f"<EVENT {rendered_attributes}>\n{_validate_content(event.content)}</EVENT>\n"


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


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


def _render_attribute_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "|".join(str(part) for part in value)
    return str(value)


def _escape_tag_attribute(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError("event tag attributes must not contain whitespace")
    return _escape_attribute(value)


def _validate_content(content: str) -> str:
    if "</EVENT>" in content:
        raise ValueError("event content must not contain </EVENT>")
    return content if content.endswith("\n") else f"{content}\n"
