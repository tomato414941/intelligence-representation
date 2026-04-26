from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BeliefStatus = Literal["active", "retired", "uncertain"]
ConflictStatus = Literal["unresolved", "resolved"]
UpdateType = Literal["add", "merge", "contradict", "retire"]


@dataclass(frozen=True)
class Observation:
    id: str
    payload: dict[str, Any]


@dataclass
class Belief:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    status: BeliefStatus
    evidence: list[str] = field(default_factory=list)
    counterevidence: list[str] = field(default_factory=list)
    scope: str = "world"

    def key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.predicate, self.object, self.scope)

    def render(self) -> str:
        return (
            f"{self.predicate}({self.subject}, {self.object}) "
            f"scope={self.scope} confidence={self.confidence:.2f} status={self.status} "
            f"evidence={self.evidence} counterevidence={self.counterevidence}"
        )


@dataclass
class Conflict:
    id: str
    belief_a: str
    belief_b: str
    type: str
    status: ConflictStatus
    possible_resolutions: list[str]


@dataclass(frozen=True)
class UpdateLogEntry:
    id: str
    type: UpdateType
    observation: str
    target: str | None = None
    result: str | None = None


class SemanticMemory:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.beliefs: list[Belief] = []
        self.conflicts: list[Conflict] = []
        self.update_log: list[UpdateLogEntry] = []
        self._next_observation_id = 1
        self._next_belief_id = 1
        self._next_conflict_id = 1
        self._next_update_id = 1

    def process_claim(self, payload: dict[str, Any]) -> Belief:
        observation = self._save_observation(payload)
        existing = self._find_same_belief(payload)
        if existing:
            existing.evidence.append(observation.id)
            existing.confidence = min(1.0, existing.confidence + 0.05)
            self._log("merge", observation=observation.id, target=existing.id, result=existing.id)
            return existing

        belief = self._create_belief(payload, observation.id)
        conflicts = self._find_conflicting_beliefs(belief)
        if conflicts:
            belief.status = "uncertain"
            for conflict_target in conflicts:
                conflict_target.counterevidence.append(observation.id)
                self._create_conflict(conflict_target, belief)
                self._log("contradict", observation=observation.id, target=conflict_target.id, result=belief.id)
        else:
            self._log("add", observation=observation.id, result=belief.id)
        return belief

    def retire_belief(self, belief_id: str, payload: dict[str, Any]) -> None:
        observation = self._save_observation(payload)
        belief = self._find_belief_by_id(belief_id)
        belief.status = "retired"
        self._log("retire", observation=observation.id, target=belief.id, result=belief.id)

    def _save_observation(self, payload: dict[str, Any]) -> Observation:
        observation = Observation(id=f"obs_{self._next_observation_id}", payload=dict(payload))
        self._next_observation_id += 1
        self.observations.append(observation)
        return observation

    def _create_belief(self, payload: dict[str, Any], observation_id: str) -> Belief:
        belief = Belief(
            id=f"belief_{self._next_belief_id}",
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            confidence=payload.get("confidence", 0.8),
            status="active",
            evidence=[observation_id],
            scope=payload.get("scope", "world"),
        )
        self._next_belief_id += 1
        self.beliefs.append(belief)
        return belief

    def _create_conflict(self, belief_a: Belief, belief_b: Belief) -> Conflict:
        conflict = Conflict(
            id=f"conflict_{self._next_conflict_id}",
            belief_a=belief_a.id,
            belief_b=belief_b.id,
            type="exclusive_has",
            status="unresolved",
            possible_resolutions=[
                "different_context",
                "newer_information_overrides_older",
                "one_observation_is_wrong",
                "extraction_error",
            ],
        )
        self._next_conflict_id += 1
        self.conflicts.append(conflict)
        return conflict

    def _log(
        self,
        update_type: UpdateType,
        *,
        observation: str,
        target: str | None = None,
        result: str | None = None,
    ) -> None:
        self.update_log.append(
            UpdateLogEntry(
                id=f"update_{self._next_update_id}",
                type=update_type,
                observation=observation,
                target=target,
                result=result,
            )
        )
        self._next_update_id += 1

    def _find_same_belief(self, payload: dict[str, Any]) -> Belief | None:
        key = (
            payload["subject"],
            payload["predicate"],
            payload["object"],
            payload.get("scope", "world"),
        )
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
            and existing.scope == belief.scope
        ]

    def _find_belief_by_id(self, belief_id: str) -> Belief:
        for belief in self.beliefs:
            if belief.id == belief_id:
                return belief
        raise ValueError(f"Unknown belief: {belief_id}")


def run_demo() -> None:
    memory = SemanticMemory()
    memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})
    memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})
    memory.process_claim({"subject": "佐藤", "predicate": "has", "object": "本"})

    print("Beliefs:")
    for belief in memory.beliefs:
        print(f"- {belief.render()}")

    print()
    print("Conflicts:")
    for conflict in memory.conflicts:
        print(f"- {conflict.id}: {conflict.belief_a} vs {conflict.belief_b} status={conflict.status}")

    print()
    print("Update log:")
    for entry in memory.update_log:
        print(f"- {entry.id}: {entry.type} observation={entry.observation} target={entry.target} result={entry.result}")


if __name__ == "__main__":
    run_demo()
