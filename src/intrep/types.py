from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


PredictionStatus = Literal["pending", "confirmed", "mismatch", "unsupported"]
ErrorType = Literal["none", "mismatch", "unsupported"]


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def render(self) -> str:
        return f"{self.predicate}({self.subject}, {self.object})"


@dataclass(frozen=True)
class Action:
    type: str
    actor: str
    object: str
    target: str


class Predictor(Protocol):
    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        ...

