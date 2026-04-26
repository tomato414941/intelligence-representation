from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BeliefStatus = Literal["active", "retired", "uncertain"]
ConflictStatus = Literal["unresolved", "resolved"]
UpdateType = Literal["add_claim", "merge_belief", "create_conflict", "retire_belief"]


@dataclass(frozen=True)
class Observation:
    id: str
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
    time: str
    context: str
    owner_of_belief: str
    confidence: float

    def belief_key(self) -> tuple[str, str, str, str, str]:
        return (self.subject, self.predicate, self.object, self.context, self.owner_of_belief)


@dataclass
class Belief:
    id: str
    subject: str
    predicate: str
    object: str
    context: str
    owner_of_belief: str
    status: BeliefStatus = "active"
    confidence: float = 0.0
    supporting_claims: list[str] = field(default_factory=list)
    counter_claims: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str, str, str, str]:
        return (self.subject, self.predicate, self.object, self.context, self.owner_of_belief)

    def render(self) -> str:
        return (
            f"{self.predicate}({self.subject}, {self.object}) "
            f"context={self.context} owner={self.owner_of_belief} "
            f"confidence={self.confidence:.2f} status={self.status} "
            f"support={self.supporting_claims} counter={self.counter_claims}"
        )


@dataclass
class Conflict:
    id: str
    left: str
    right: str
    type: str
    status: ConflictStatus
    possible_resolutions: list[str]


@dataclass(frozen=True)
class UpdateLogEntry:
    id: str
    type: UpdateType
    observation_id: str | None = None
    claim_id: str | None = None
    target_id: str | None = None
    result_id: str | None = None


class SemanticMemory:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.claims: list[Claim] = []
        self.beliefs: list[Belief] = []
        self.conflicts: list[Conflict] = []
        self.update_log: list[UpdateLogEntry] = []
        self._next_observation_id = 1
        self._next_claim_id = 1
        self._next_belief_id = 1
        self._next_conflict_id = 1
        self._next_update_id = 1

    def ingest(self, payload: dict[str, Any], *, source: str = "manual", created_at: str = "t0") -> Claim:
        observation = self._add_observation(payload, source=source, created_at=created_at)
        claim = self._add_claim(observation)
        self._log("add_claim", observation_id=observation.id, claim_id=claim.id, result_id=claim.id)
        self._integrate_claim(claim, observation.id)
        return claim

    def retire_belief(self, belief_id: str, *, reason: str = "manual") -> None:
        belief = self._belief_by_id(belief_id)
        belief.status = "retired"
        self._log("retire_belief", target_id=belief.id, result_id=belief.id)

    def _add_observation(self, payload: dict[str, Any], *, source: str, created_at: str) -> Observation:
        observation = Observation(
            id=f"obs_{self._next_observation_id}",
            payload=dict(payload),
            source=source,
            created_at=created_at,
        )
        self._next_observation_id += 1
        self.observations.append(observation)
        return observation

    def _add_claim(self, observation: Observation) -> Claim:
        payload = observation.payload
        claim = Claim(
            id=f"claim_{self._next_claim_id}",
            observation_id=observation.id,
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            time=payload.get("time", observation.created_at),
            context=payload.get("context", "world"),
            owner_of_belief=payload.get("owner_of_belief", "world"),
            confidence=payload.get("confidence", 0.8),
        )
        self._next_claim_id += 1
        self.claims.append(claim)
        return claim

    def _integrate_claim(self, claim: Claim, observation_id: str) -> None:
        belief = self._find_matching_belief(claim)
        if belief:
            belief.supporting_claims.append(claim.id)
            belief.confidence = min(1.0, belief.confidence + 0.05)
            self._log(
                "merge_belief",
                observation_id=observation_id,
                claim_id=claim.id,
                target_id=belief.id,
                result_id=belief.id,
            )
            return

        belief = self._create_belief_from_claim(claim)
        conflicts = self._find_conflicting_beliefs(claim, belief)
        if conflicts:
            belief.status = "uncertain"
            for existing in conflicts:
                existing.counter_claims.append(claim.id)
                belief.counter_claims.append(existing.supporting_claims[-1])
                conflict = self._create_conflict(existing, belief)
                self._log(
                    "create_conflict",
                    observation_id=observation_id,
                    claim_id=claim.id,
                    target_id=existing.id,
                    result_id=conflict.id,
                )

    def _create_belief_from_claim(self, claim: Claim) -> Belief:
        belief = Belief(
            id=f"belief_{self._next_belief_id}",
            subject=claim.subject,
            predicate=claim.predicate,
            object=claim.object,
            context=claim.context,
            owner_of_belief=claim.owner_of_belief,
            confidence=claim.confidence,
            supporting_claims=[claim.id],
        )
        self._next_belief_id += 1
        self.beliefs.append(belief)
        return belief

    def _create_conflict(self, left: Belief, right: Belief) -> Conflict:
        conflict = Conflict(
            id=f"conflict_{self._next_conflict_id}",
            left=left.id,
            right=right.id,
            type="exclusive_has",
            status="unresolved",
            possible_resolutions=[
                "different_context",
                "different_time",
                "one_claim_is_wrong",
                "extraction_error",
            ],
        )
        self._next_conflict_id += 1
        self.conflicts.append(conflict)
        return conflict

    def _find_matching_belief(self, claim: Claim) -> Belief | None:
        for belief in self.beliefs:
            if belief.status == "active" and belief.key() == claim.belief_key():
                return belief
        return None

    def _find_conflicting_beliefs(self, claim: Claim, new_belief: Belief) -> list[Belief]:
        if claim.predicate != "has":
            return []
        return [
            belief
            for belief in self.beliefs
            if belief.id != new_belief.id
            and belief.status == "active"
            and belief.predicate == "has"
            and belief.object == claim.object
            and belief.subject != claim.subject
            and belief.context == claim.context
            and belief.owner_of_belief == claim.owner_of_belief
        ]

    def _belief_by_id(self, belief_id: str) -> Belief:
        for belief in self.beliefs:
            if belief.id == belief_id:
                return belief
        raise ValueError(f"Unknown belief: {belief_id}")

    def _log(
        self,
        update_type: UpdateType,
        *,
        observation_id: str | None = None,
        claim_id: str | None = None,
        target_id: str | None = None,
        result_id: str | None = None,
    ) -> None:
        entry = UpdateLogEntry(
            id=f"update_{self._next_update_id}",
            type=update_type,
            observation_id=observation_id,
            claim_id=claim_id,
            target_id=target_id,
            result_id=result_id,
        )
        self._next_update_id += 1
        self.update_log.append(entry)


def run_demo() -> None:
    memory = SemanticMemory()
    memory.ingest({"subject": "田中", "predicate": "has", "object": "本", "time": "t1"})
    memory.ingest({"subject": "田中", "predicate": "has", "object": "本", "time": "t1"})
    memory.ingest({"subject": "佐藤", "predicate": "has", "object": "本", "time": "t1"})

    print("Beliefs:")
    for belief in memory.beliefs:
        print(f"- {belief.render()}")

    print()
    print("Conflicts:")
    for conflict in memory.conflicts:
        print(f"- {conflict.id}: {conflict.left} vs {conflict.right} status={conflict.status}")


if __name__ == "__main__":
    run_demo()
