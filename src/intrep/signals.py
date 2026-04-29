from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlparse


@dataclass(frozen=True)
class PayloadRef:
    uri: str
    media_type: str
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("payload ref uri must be a non-empty string")
        if any(character.isspace() for character in self.uri):
            raise ValueError("payload ref uri must not contain whitespace")
        parsed = urlparse(self.uri)
        if not parsed.scheme:
            raise ValueError("payload ref uri must include a scheme")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("payload ref media_type must be a non-empty string")
        if "/" not in self.media_type or any(character.isspace() for character in self.media_type):
            raise ValueError("payload ref media_type must look like a MIME type")
        if self.sha256 is not None:
            if not isinstance(self.sha256, str) or len(self.sha256) != 64:
                raise ValueError("payload ref sha256 must be a 64-character hex string")
            try:
                int(self.sha256, 16)
            except ValueError as error:
                raise ValueError("payload ref sha256 must be a 64-character hex string") from error
        if self.size_bytes is not None:
            if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
                raise ValueError("payload ref size_bytes must be a non-negative integer")


SignalPayload: TypeAlias = str | PayloadRef


@dataclass(frozen=True)
class Signal:
    channel: str
    payload: SignalPayload

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("signal channel must be a non-empty string")
        if any(character.isspace() for character in self.channel):
            raise ValueError("signal channel must not contain whitespace")
        if not isinstance(self.payload, (str, PayloadRef)):
            raise ValueError("signal payload must be a string or PayloadRef")


def render_signal(signal: Signal) -> str:
    attributes = [
        ("channel", signal.channel),
    ]
    rendered_attributes = " ".join(
        f'{name}="{_escape_tag_attribute(value)}"' for name, value in attributes
    )
    return f"<SIGNAL {rendered_attributes}>\n{_validate_payload(render_payload_text(signal))}</SIGNAL>\n"


def require_text_payload(signal: Signal, *, context: str = "signal rendering") -> str:
    if not isinstance(signal.payload, str):
        raise ValueError(f"{context} requires a string signal payload")
    return signal.payload


def render_payload_text(signal: Signal) -> str:
    if isinstance(signal.payload, str):
        return signal.payload
    raise ValueError("payload ref requires a channel-specific loader or encoder")


def _escape_attribute(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_tag_attribute(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError("signal tag attributes must not contain whitespace")
    return _escape_attribute(value)


def _validate_payload(payload: str) -> str:
    if "</SIGNAL>" in payload:
        raise ValueError("signal payload must not contain </SIGNAL>")
    return payload if payload.endswith("\n") else f"{payload}\n"
