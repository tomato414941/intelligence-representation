from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ClaimStatus = Literal["active", "inactive", "uncertain", "contradicted", "superseded"]


@dataclass
class Claim:
    id: str
    subject: str
    predicate: str
    object: str
    source: str
    time: str
    context: str
    owner_of_belief: str = "world"
    confidence: float = 0.9
    status: ClaimStatus = "active"
    invalidated_by: str | None = None
    conflicts_with: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)

    def render(self) -> str:
        extras = []
        if self.invalidated_by:
            extras.append(f"invalidated_by={self.invalidated_by}")
        if self.conflicts_with:
            extras.append(f"conflicts_with={self.conflicts_with}")
        extra_text = f" {' '.join(extras)}" if extras else ""
        return (
            f"{self.predicate}({self.subject}, {self.object}) "
            f"time={self.time} context={self.context} owner={self.owner_of_belief} "
            f"status={self.status}{extra_text}"
        )


@dataclass(frozen=True)
class Observation:
    id: str
    event: dict[str, Any]


class ContextualState:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.claims: list[Claim] = []
        self._next_observation_id = 1
        self._next_claim_id = 1

    def observe(self, event: dict[str, Any]) -> None:
        observation = Observation(id=f"obs_{self._next_observation_id}", event=event)
        self._next_observation_id += 1
        self.observations.append(observation)

        event_type = event["type"]
        if event_type == "claim":
            self._add_claim_from_event(observation, event)
            return
        if event_type == "transfer":
            self._deactivate_latest(
                subject=event["actor"],
                predicate="has",
                object=event["object"],
                context=event["context"],
                owner_of_belief=event.get("owner_of_belief", "world"),
                invalidated_by=observation.id,
            )
            self._add_claim(
                subject=event["recipient"],
                predicate="has",
                object=event["object"],
                source=observation.id,
                time=event["time"],
                context=event["context"],
                owner_of_belief=event.get("owner_of_belief", "world"),
                confidence=event.get("confidence", 0.9),
            )
            return
        if event_type == "place":
            self._deactivate_latest(
                subject=event["actor"],
                predicate="has",
                object=event["object"],
                context=event["context"],
                owner_of_belief=event.get("owner_of_belief", "world"),
                invalidated_by=observation.id,
            )
            self._add_claim(
                subject=event["object"],
                predicate="located_at",
                object=event["location"],
                source=observation.id,
                time=event["time"],
                context=event["context"],
                owner_of_belief=event.get("owner_of_belief", "world"),
                confidence=event.get("confidence", 0.9),
            )
            return

        raise ValueError(f"Unsupported event type: {event_type}")

    def active_claims(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.status == "active"]

    def all_claims(self) -> list[Claim]:
        return list(self.claims)

    def _add_claim_from_event(self, observation: Observation, event: dict[str, Any]) -> Claim:
        return self._add_claim(
            subject=event["subject"],
            predicate=event["predicate"],
            object=event["object"],
            source=observation.id,
            time=event["time"],
            context=event["context"],
            owner_of_belief=event.get("owner_of_belief", "world"),
            confidence=event.get("confidence", 0.9),
        )

    def _add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        source: str,
        time: str,
        context: str,
        owner_of_belief: str,
        confidence: float,
    ) -> Claim:
        conflicts = self._find_conflicts(
            subject=subject,
            predicate=predicate,
            object=object,
            time=time,
            context=context,
            owner_of_belief=owner_of_belief,
        )
        status: ClaimStatus = "uncertain" if conflicts else "active"
        claim = Claim(
            id=f"claim_{self._next_claim_id}",
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            time=time,
            context=context,
            owner_of_belief=owner_of_belief,
            confidence=confidence,
            status=status,
            conflicts_with=[conflict.id for conflict in conflicts],
        )
        self._next_claim_id += 1
        self.claims.append(claim)
        return claim

    def _deactivate_latest(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        context: str,
        owner_of_belief: str,
        invalidated_by: str,
    ) -> None:
        for claim in reversed(self.claims):
            if (
                claim.status == "active"
                and claim.subject == subject
                and claim.predicate == predicate
                and claim.object == object
                and claim.context == context
                and claim.owner_of_belief == owner_of_belief
            ):
                claim.status = "inactive"
                claim.invalidated_by = invalidated_by
                return

    def _find_conflicts(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        time: str,
        context: str,
        owner_of_belief: str,
    ) -> list[Claim]:
        if predicate != "has":
            return []
        return [
            claim
            for claim in self.claims
            if claim.status == "active"
            and claim.predicate == "has"
            and claim.object == object
            and claim.subject != subject
            and claim.time == time
            and claim.context == context
            and claim.owner_of_belief == owner_of_belief
        ]


def run_demo() -> None:
    state = ContextualState()
    events = [
        {"type": "claim", "subject": "田中", "predicate": "has", "object": "本", "time": "t1", "context": "world"},
        {"type": "transfer", "actor": "田中", "recipient": "佐藤", "object": "本", "time": "t2", "context": "world"},
        {"type": "place", "actor": "佐藤", "object": "本", "location": "図書館", "time": "t3", "context": "world"},
    ]
    for event in events:
        state.observe(event)
    for claim in state.all_claims():
        print(f"- {claim.render()}")


if __name__ == "__main__":
    run_demo()
