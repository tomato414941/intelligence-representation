from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


EventType = Literal["claim", "transfer", "place"]


@dataclass(frozen=True)
class Observation:
    id: str
    event: dict[str, Any]


@dataclass
class Claim:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    active: bool = True
    invalidated_by: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def render(self) -> str:
        status = "active" if self.active else "inactive"
        return f"{self.predicate}({self.subject}, {self.object}): {status}"


@dataclass
class StateUpdate:
    source: str
    deactivate: list[tuple[str, str, str]] = field(default_factory=list)
    add: list[tuple[str, str, str, float]] = field(default_factory=list)


class SemanticState:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.claims: list[Claim] = []
        self._next_observation_id = 1
        self._next_claim_id = 1

    def observe(self, event: dict[str, Any]) -> StateUpdate:
        observation = Observation(id=f"obs_{self._next_observation_id}", event=event)
        self._next_observation_id += 1
        self.observations.append(observation)

        update = self._build_update(observation)
        self.apply(update)
        return update

    def apply(self, update: StateUpdate) -> None:
        for key in update.deactivate:
            for claim in self._active_claims_by_key(key):
                claim.active = False
                claim.invalidated_by = update.source

        for subject, predicate, object_, confidence in update.add:
            claim = Claim(
                id=f"claim_{self._next_claim_id}",
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence,
                source=update.source,
            )
            self._next_claim_id += 1
            self.claims.append(claim)

    def active_claims(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.active]

    def all_claims(self) -> list[Claim]:
        return list(self.claims)

    def answer_location(self, object_: str) -> str:
        for claim in reversed(self.active_claims()):
            if claim.subject == object_ and claim.predicate == "located_at":
                return f"{object_}は{claim.object}にある可能性が高い"
        for claim in reversed(self.active_claims()):
            if claim.object == object_ and claim.predicate == "has":
                return f"{object_}は{claim.subject}が持っている可能性が高い"
        return f"{object_}の場所は不明"

    def _active_claims_by_key(self, key: tuple[str, str, str]) -> Iterable[Claim]:
        return (claim for claim in self.claims if claim.active and claim.key() == key)

    def _build_update(self, observation: Observation) -> StateUpdate:
        event = observation.event
        event_type: EventType = event["type"]

        if event_type == "claim":
            return StateUpdate(
                source=observation.id,
                add=[
                    (
                        event["subject"],
                        event["predicate"],
                        event["object"],
                        event.get("confidence", 0.9),
                    )
                ],
            )

        if event_type == "transfer":
            return StateUpdate(
                source=observation.id,
                deactivate=[(event["actor"], "has", event["object"])],
                add=[(event["recipient"], "has", event["object"], event.get("confidence", 0.9))],
            )

        if event_type == "place":
            return StateUpdate(
                source=observation.id,
                deactivate=[(event["actor"], "has", event["object"])],
                add=[(event["object"], "located_at", event["location"], event.get("confidence", 0.9))],
            )

        raise ValueError(f"Unsupported event type: {event_type}")


def run_demo() -> None:
    state = SemanticState()
    events = [
        {"type": "claim", "subject": "田中", "predicate": "has", "object": "本"},
        {"type": "transfer", "actor": "田中", "recipient": "佐藤", "object": "本"},
        {"type": "place", "actor": "佐藤", "object": "本", "location": "図書館"},
    ]

    for event in events:
        state.observe(event)

    print("All claims:")
    for claim in state.all_claims():
        print(f"- {claim.render()} source={claim.source}")

    print()
    print("Active claims:")
    for claim in state.active_claims():
        print(f"- {claim.render()} source={claim.source}")

    print()
    print(state.answer_location("本"))


if __name__ == "__main__":
    run_demo()
