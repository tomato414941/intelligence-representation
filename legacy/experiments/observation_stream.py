from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Modality = Literal[
    "text",
    "image",
    "audio",
    "video",
    "sensor",
    "action_result",
    "tool_output",
    "environment_state",
]


@dataclass(frozen=True)
class Observation:
    id: str
    modality: Modality
    payload: dict[str, Any]
    source: str
    created_at: str


@dataclass(frozen=True)
class Claim:
    id: str
    observation_id: str
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(frozen=True)
class Event:
    id: str
    observation_id: str
    type: str
    participants: dict[str, str]
    time: str
    context: str


@dataclass(frozen=True)
class StateTransition:
    id: str
    observation_id: str
    before: dict[str, list[str]]
    after: dict[str, list[str]]
    cause: str
    confidence: float


@dataclass
class Belief:
    id: str
    subject: str
    predicate: str
    object: str
    supporting_claims: list[str] = field(default_factory=list)


class ObservationStreamMemory:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.claims: list[Claim] = []
        self.events: list[Event] = []
        self.state_transitions: list[StateTransition] = []
        self.beliefs: list[Belief] = []
        self._next_observation_id = 1
        self._next_claim_id = 1
        self._next_event_id = 1
        self._next_transition_id = 1
        self._next_belief_id = 1

    def ingest(
        self,
        *,
        modality: Modality,
        payload: dict[str, Any],
        source: str = "manual",
        created_at: str = "t0",
    ) -> Observation:
        observation = self._add_observation(
            modality=modality,
            payload=payload,
            source=source,
            created_at=created_at,
        )

        if modality == "text":
            claim = self._claim_from_text_observation(observation)
            self._merge_claim_into_belief(claim)
            return observation

        if modality == "action_result":
            event = self._event_from_action_result(observation)
            self._state_transition_from_action_result(observation, cause=event.id)
            return observation

        return observation

    def _add_observation(
        self,
        *,
        modality: Modality,
        payload: dict[str, Any],
        source: str,
        created_at: str,
    ) -> Observation:
        observation = Observation(
            id=f"obs_{self._next_observation_id}",
            modality=modality,
            payload=dict(payload),
            source=source,
            created_at=created_at,
        )
        self._next_observation_id += 1
        self.observations.append(observation)
        return observation

    def _claim_from_text_observation(self, observation: Observation) -> Claim:
        payload = observation.payload
        claim = Claim(
            id=f"claim_{self._next_claim_id}",
            observation_id=observation.id,
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            confidence=payload.get("confidence", 0.8),
        )
        self._next_claim_id += 1
        self.claims.append(claim)
        return claim

    def _event_from_action_result(self, observation: Observation) -> Event:
        payload = observation.payload
        participants = {
            key: value
            for key, value in payload.items()
            if key in {"actor", "object", "recipient", "location"}
        }
        event = Event(
            id=f"event_{self._next_event_id}",
            observation_id=observation.id,
            type=payload["event_type"],
            participants=participants,
            time=payload.get("time", observation.created_at),
            context=payload.get("context", "world"),
        )
        self._next_event_id += 1
        self.events.append(event)
        return event

    def _state_transition_from_action_result(self, observation: Observation, *, cause: str) -> StateTransition:
        payload = observation.payload
        transition = StateTransition(
            id=f"transition_{self._next_transition_id}",
            observation_id=observation.id,
            before=payload.get("before", {}),
            after=payload.get("after", {}),
            cause=cause,
            confidence=payload.get("confidence", 0.8),
        )
        self._next_transition_id += 1
        self.state_transitions.append(transition)
        return transition

    def _merge_claim_into_belief(self, claim: Claim) -> Belief:
        for belief in self.beliefs:
            if (
                belief.subject == claim.subject
                and belief.predicate == claim.predicate
                and belief.object == claim.object
            ):
                belief.supporting_claims.append(claim.id)
                return belief

        belief = Belief(
            id=f"belief_{self._next_belief_id}",
            subject=claim.subject,
            predicate=claim.predicate,
            object=claim.object,
            supporting_claims=[claim.id],
        )
        self._next_belief_id += 1
        self.beliefs.append(belief)
        return belief


def run_demo() -> None:
    memory = ObservationStreamMemory()
    memory.ingest(
        modality="text",
        payload={"subject": "田中", "predicate": "has", "object": "本"},
    )
    memory.ingest(
        modality="action_result",
        payload={
            "event_type": "place",
            "actor": "佐藤",
            "object": "本",
            "location": "図書館",
            "before": {"has": ["佐藤", "本"]},
            "after": {"located_at": ["本", "図書館"]},
        },
    )

    print(f"observations={len(memory.observations)}")
    print(f"claims={len(memory.claims)}")
    print(f"events={len(memory.events)}")
    print(f"state_transitions={len(memory.state_transitions)}")
    print(f"beliefs={len(memory.beliefs)}")


if __name__ == "__main__":
    run_demo()
