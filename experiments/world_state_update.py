from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FactStatus = Literal["active", "inactive"]


@dataclass
class Fact:
    id: str
    subject: str
    predicate: str
    object: str
    source: str
    status: FactStatus = "active"
    invalidated_by: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def render(self) -> str:
        suffix = f" invalidated_by={self.invalidated_by}" if self.invalidated_by else ""
        return f"{self.predicate}({self.subject}, {self.object}): {self.status}{suffix}"


@dataclass(frozen=True)
class StateTransition:
    id: str
    before: dict[str, list[str]]
    after: dict[str, list[str]]
    source: str


class WorldState:
    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.transitions: list[StateTransition] = []
        self._next_fact_id = 1
        self._next_transition_id = 1

    def add_fact(self, predicate: str, subject: str, object: str, *, source: str = "manual") -> Fact:
        fact = Fact(
            id=f"fact_{self._next_fact_id}",
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
        )
        self._next_fact_id += 1
        self.facts.append(fact)
        return fact

    def apply_transition(
        self,
        *,
        before: dict[str, list[str]],
        after: dict[str, list[str]],
        source: str,
    ) -> StateTransition:
        transition = StateTransition(
            id=f"transition_{self._next_transition_id}",
            before=before,
            after=after,
            source=source,
        )
        self._next_transition_id += 1
        self.transitions.append(transition)

        for predicate, args in before.items():
            self._deactivate_fact(predicate, args[0], args[1], invalidated_by=transition.id)

        for predicate, args in after.items():
            self.add_fact(predicate, args[0], args[1], source=transition.id)

        return transition

    def active_facts(self) -> list[Fact]:
        return [fact for fact in self.facts if fact.status == "active"]

    def inactive_facts(self) -> list[Fact]:
        return [fact for fact in self.facts if fact.status == "inactive"]

    def _deactivate_fact(
        self,
        predicate: str,
        subject: str,
        object: str,
        *,
        invalidated_by: str,
    ) -> None:
        key = (subject, predicate, object)
        for fact in reversed(self.facts):
            if fact.status == "active" and fact.key() == key:
                fact.status = "inactive"
                fact.invalidated_by = invalidated_by
                return


def run_demo() -> None:
    world = WorldState()
    world.add_fact("has", "佐藤", "本")
    world.apply_transition(
        before={"has": ["佐藤", "本"]},
        after={"located_at": ["本", "図書館"]},
        source="obs_1",
    )

    for fact in world.facts:
        print(f"- {fact.render()}")


if __name__ == "__main__":
    run_demo()
