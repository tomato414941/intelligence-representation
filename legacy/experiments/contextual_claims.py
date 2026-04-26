from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ClaimStatus = Literal["active", "inactive", "uncertain", "contradicted", "superseded"]


@dataclass
class ContextualClaim:
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
    conflicts_with: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)

    def render(self) -> str:
        conflict = f" conflicts_with={self.conflicts_with}" if self.conflicts_with else ""
        return (
            f"{self.predicate}({self.subject}, {self.object}) "
            f"time={self.time} context={self.context} owner={self.owner_of_belief} "
            f"status={self.status}{conflict}"
        )


class ContextualClaimState:
    def __init__(self) -> None:
        self.claims: list[ContextualClaim] = []
        self._next_claim_id = 1

    def add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        source: str,
        time: str,
        context: str,
        owner_of_belief: str = "world",
        confidence: float = 0.9,
    ) -> ContextualClaim:
        conflicts = self._find_conflicts(
            subject=subject,
            predicate=predicate,
            object=object,
            time=time,
            context=context,
            owner_of_belief=owner_of_belief,
        )
        status: ClaimStatus = "uncertain" if conflicts else "active"

        claim = ContextualClaim(
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

    def active_claims(self) -> list[ContextualClaim]:
        return [claim for claim in self.claims if claim.status == "active"]

    def uncertain_claims(self) -> list[ContextualClaim]:
        return [claim for claim in self.claims if claim.status == "uncertain"]

    def all_claims(self) -> list[ContextualClaim]:
        return list(self.claims)

    def _find_conflicts(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        time: str,
        context: str,
        owner_of_belief: str,
    ) -> list[ContextualClaim]:
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
    state = ContextualClaimState()
    state.add_claim(
        subject="田中",
        predicate="has",
        object="本",
        source="obs_1",
        time="t1",
        context="world",
    )
    state.add_claim(
        subject="佐藤",
        predicate="has",
        object="本",
        source="obs_2",
        time="t1",
        context="world",
    )
    state.add_claim(
        subject="佐藤",
        predicate="has",
        object="本",
        source="obs_3",
        time="t2",
        context="world",
    )

    for claim in state.all_claims():
        print(f"- {claim.render()}")


if __name__ == "__main__":
    run_demo()
