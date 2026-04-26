from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from intrep.evaluation import PredictionCase
from intrep.types import Action, Fact


@dataclass(frozen=True)
class ActionConditionedExample:
    id: str
    state_before: list[Fact]
    action: Action
    expected_observation: Fact | None = None
    expected_state_after: list[Fact] = field(default_factory=list)
    source: str = "manual"

    def to_prediction_case(self) -> PredictionCase:
        expected = self.expected_observation
        if expected is None and len(self.expected_state_after) == 1:
            expected = self.expected_state_after[0]
        return PredictionCase(
            name=self.id,
            initial_state=self.state_before,
            action=self.action,
            expected_fact=expected,
        )


def fact_to_dict(fact: Fact) -> dict[str, str]:
    return {"subject": fact.subject, "predicate": fact.predicate, "object": fact.object}


def fact_from_dict(data: dict[str, str]) -> Fact:
    return Fact(subject=data["subject"], predicate=data["predicate"], object=data["object"])


def action_to_dict(action: Action) -> dict[str, str]:
    return {"type": action.type, "actor": action.actor, "object": action.object, "target": action.target}


def action_from_dict(data: dict[str, str]) -> Action:
    return Action(type=data["type"], actor=data["actor"], object=data["object"], target=data["target"])


def example_to_dict(example: ActionConditionedExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "state_before": [fact_to_dict(fact) for fact in example.state_before],
        "action": action_to_dict(example.action),
        "expected_observation": fact_to_dict(example.expected_observation)
        if example.expected_observation
        else None,
        "expected_state_after": [fact_to_dict(fact) for fact in example.expected_state_after],
        "source": example.source,
    }


def example_from_dict(data: dict[str, Any]) -> ActionConditionedExample:
    expected_observation = data.get("expected_observation")
    return ActionConditionedExample(
        id=data["id"],
        state_before=[fact_from_dict(fact) for fact in data.get("state_before", [])],
        action=action_from_dict(data["action"]),
        expected_observation=fact_from_dict(expected_observation) if expected_observation else None,
        expected_state_after=[fact_from_dict(fact) for fact in data.get("expected_state_after", [])],
        source=data.get("source", "manual"),
    )


def dumps_jsonl(examples: list[ActionConditionedExample]) -> str:
    return "\n".join(json.dumps(example_to_dict(example), ensure_ascii=False) for example in examples)


def loads_jsonl(content: str) -> list[ActionConditionedExample]:
    examples = []
    for line in content.splitlines():
        if line.strip():
            examples.append(example_from_dict(json.loads(line)))
    return examples

