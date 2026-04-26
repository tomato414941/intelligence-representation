from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

try:
    from experiments.predictor_evaluation import PredictionCase
    from experiments.predictor_interface import Action, Fact
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.predictor_evaluation import PredictionCase
    from experiments.predictor_interface import Action, Fact


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


def smoke_examples() -> list[ActionConditionedExample]:
    return [
        ActionConditionedExample(
            id="place_book_library",
            state_before=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="place", actor="佐藤", object="本", target="図書館"),
            expected_observation=Fact(subject="本", predicate="located_at", object="図書館"),
            expected_state_after=[
                Fact(subject="佐藤", predicate="has", object="本"),
                Fact(subject="本", predicate="located_at", object="図書館"),
            ],
        ),
        ActionConditionedExample(
            id="unsupported_throw_book",
            state_before=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="throw", actor="佐藤", object="本", target="床"),
            expected_observation=None,
            expected_state_after=[],
        ),
    ]


def run_demo() -> None:
    content = dumps_jsonl(smoke_examples())
    loaded = loads_jsonl(content)
    for example in loaded:
        expected = example.expected_observation.render() if example.expected_observation else "unsupported"
        print(f"{example.id}: action={example.action.type} expected={expected}")


if __name__ == "__main__":
    run_demo()
