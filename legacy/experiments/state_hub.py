from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Modality = Literal["text", "action_result", "sensor", "tool_output"]
FactStatus = Literal["active", "inactive"]
BeliefStatus = Literal["active", "uncertain", "retired"]


@dataclass(frozen=True)
class Observation:
    id: str
    modality: Modality
    payload: dict[str, Any]
    source: str
    created_at: str


@dataclass
class Fact:
    id: str
    subject: str
    predicate: str
    object: str
    source_observation: str
    status: FactStatus = "active"
    invalidated_by: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass
class Belief:
    id: str
    subject: str
    predicate: str
    object: str
    status: BeliefStatus = "active"
    supporting_observations: list[str] = field(default_factory=list)
    counter_observations: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass(frozen=True)
class Conflict:
    id: str
    left: str
    right: str
    type: str
    source_observation: str
    status: str = "unresolved"


@dataclass(frozen=True)
class Provenance:
    observation_id: str
    target_type: str
    target_id: str
    relation: str


class StateHub:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.world_facts: list[Fact] = []
        self.beliefs: list[Belief] = []
        self.conflicts: list[Conflict] = []
        self.provenance: list[Provenance] = []
        self._next_observation_id = 1
        self._next_fact_id = 1
        self._next_belief_id = 1
        self._next_conflict_id = 1

    def observe(
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
            self._update_belief_state(observation)
        elif modality == "action_result":
            self._update_world_state(observation)

        return observation

    def add_world_fact(self, predicate: str, subject: str, object: str, *, source_observation: str) -> Fact:
        fact = Fact(
            id=f"fact_{self._next_fact_id}",
            subject=subject,
            predicate=predicate,
            object=object,
            source_observation=source_observation,
        )
        self._next_fact_id += 1
        self.world_facts.append(fact)
        self._add_provenance(source_observation, "fact", fact.id, "created")
        return fact

    def active_world_facts(self) -> list[Fact]:
        return [fact for fact in self.world_facts if fact.status == "active"]

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

    def _update_belief_state(self, observation: Observation) -> None:
        payload = observation.payload
        key = (payload["subject"], payload["predicate"], payload["object"])
        existing = self._find_belief_by_key(key)
        if existing:
            existing.supporting_observations.append(observation.id)
            self._add_provenance(observation.id, "belief", existing.id, "supports")
            return

        belief = Belief(
            id=f"belief_{self._next_belief_id}",
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            supporting_observations=[observation.id],
        )
        self._next_belief_id += 1
        self.beliefs.append(belief)
        self._add_provenance(observation.id, "belief", belief.id, "created")

        conflicts = self._find_conflicting_beliefs(belief)
        if conflicts:
            belief.status = "uncertain"
            for conflict_target in conflicts:
                conflict_target.counter_observations.append(observation.id)
                conflict = Conflict(
                    id=f"conflict_{self._next_conflict_id}",
                    left=conflict_target.id,
                    right=belief.id,
                    type="exclusive_has",
                    source_observation=observation.id,
                )
                self._next_conflict_id += 1
                self.conflicts.append(conflict)
                self._add_provenance(observation.id, "conflict", conflict.id, "created")

    def _update_world_state(self, observation: Observation) -> None:
        payload = observation.payload
        for predicate, args in payload.get("before", {}).items():
            self._deactivate_world_fact(predicate, args[0], args[1], invalidated_by=observation.id)
        for predicate, args in payload.get("after", {}).items():
            self.add_world_fact(predicate, args[0], args[1], source_observation=observation.id)

    def _deactivate_world_fact(
        self,
        predicate: str,
        subject: str,
        object: str,
        *,
        invalidated_by: str,
    ) -> None:
        key = (subject, predicate, object)
        for fact in reversed(self.world_facts):
            if fact.status == "active" and fact.key() == key:
                fact.status = "inactive"
                fact.invalidated_by = invalidated_by
                self._add_provenance(invalidated_by, "fact", fact.id, "invalidated")
                return

    def _find_belief_by_key(self, key: tuple[str, str, str]) -> Belief | None:
        for belief in self.beliefs:
            if belief.status == "active" and belief.key() == key:
                return belief
        return None

    def _find_conflicting_beliefs(self, belief: Belief) -> list[Belief]:
        if belief.predicate != "has":
            return []
        return [
            existing
            for existing in self.beliefs
            if existing.id != belief.id
            and existing.status == "active"
            and existing.predicate == "has"
            and existing.object == belief.object
            and existing.subject != belief.subject
        ]

    def _add_provenance(self, observation_id: str, target_type: str, target_id: str, relation: str) -> None:
        self.provenance.append(
            Provenance(
                observation_id=observation_id,
                target_type=target_type,
                target_id=target_id,
                relation=relation,
            )
        )


def run_demo() -> None:
    hub = StateHub()
    hub.observe(modality="text", payload={"subject": "田中", "predicate": "has", "object": "本"})
    hub.add_world_fact("has", "佐藤", "本", source_observation="manual")
    hub.observe(
        modality="action_result",
        payload={
            "before": {"has": ["佐藤", "本"]},
            "after": {"located_at": ["本", "図書館"]},
        },
    )
    print(f"observations={len(hub.observations)}")
    print(f"beliefs={len(hub.beliefs)}")
    print(f"world_facts={len(hub.world_facts)}")
    print(f"active_world_facts={len(hub.active_world_facts())}")
    print(f"provenance={len(hub.provenance)}")


if __name__ == "__main__":
    run_demo()
