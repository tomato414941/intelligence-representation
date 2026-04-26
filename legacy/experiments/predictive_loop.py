from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PredictionStatus = Literal["pending", "confirmed", "mismatch"]
ErrorType = Literal["none", "mismatch"]


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def render(self) -> str:
        return f"{self.predicate}({self.subject}, {self.object})"


@dataclass(frozen=True)
class Action:
    type: str
    actor: str
    object: str
    target: str


@dataclass
class Prediction:
    id: str
    action: Action
    expected: Fact
    status: PredictionStatus = "pending"


@dataclass(frozen=True)
class Observation:
    id: str
    fact: Fact


@dataclass(frozen=True)
class PredictionError:
    id: str
    prediction: str
    type: ErrorType
    expected: Fact
    observed: Fact


@dataclass(frozen=True)
class UpdateLogEntry:
    id: str
    type: str
    target: str
    detail: str


class PredictiveWorld:
    def __init__(self) -> None:
        self.state: list[Fact] = []
        self.predictions: list[Prediction] = []
        self.observations: list[Observation] = []
        self.errors: list[PredictionError] = []
        self.update_log: list[UpdateLogEntry] = []
        self._next_prediction_id = 1
        self._next_observation_id = 1
        self._next_error_id = 1
        self._next_update_id = 1

    def add_fact(self, fact: Fact) -> None:
        if fact.key() not in {existing.key() for existing in self.state}:
            self.state.append(fact)
            self._log("add_fact", fact.render(), "state")

    def predict(self, action: Action) -> Prediction:
        expected = self._predict_fact(action)
        prediction = Prediction(
            id=f"prediction_{self._next_prediction_id}",
            action=action,
            expected=expected,
        )
        self._next_prediction_id += 1
        self.predictions.append(prediction)
        self._log("predict", prediction.id, expected.render())
        return prediction

    def observe(self, fact: Fact) -> Observation:
        observation = Observation(id=f"obs_{self._next_observation_id}", fact=fact)
        self._next_observation_id += 1
        self.observations.append(observation)
        self.add_fact(fact)
        return observation

    def compare(self, prediction: Prediction, observation: Observation) -> PredictionError:
        if prediction.expected.key() == observation.fact.key():
            prediction.status = "confirmed"
            error_type: ErrorType = "none"
        else:
            prediction.status = "mismatch"
            error_type = "mismatch"

        error = PredictionError(
            id=f"error_{self._next_error_id}",
            prediction=prediction.id,
            type=error_type,
            expected=prediction.expected,
            observed=observation.fact,
        )
        self._next_error_id += 1
        self.errors.append(error)
        self._log("prediction_error", error.id, error_type)
        return error

    def _predict_fact(self, action: Action) -> Fact:
        if action.type == "place":
            return Fact(subject=action.object, predicate="located_at", object=action.target)
        raise ValueError(f"Unsupported action type: {action.type}")

    def _log(self, entry_type: str, target: str, detail: str) -> None:
        self.update_log.append(
            UpdateLogEntry(
                id=f"update_{self._next_update_id}",
                type=entry_type,
                target=target,
                detail=detail,
            )
        )
        self._next_update_id += 1


def run_demo() -> None:
    world = PredictiveWorld()
    world.add_fact(Fact(subject="佐藤", predicate="has", object="本"))
    prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
    observation = world.observe(Fact(subject="本", predicate="located_at", object="図書館"))
    error = world.compare(prediction, observation)
    print(f"prediction={prediction.expected.render()} status={prediction.status}")
    print(f"error={error.type}")


if __name__ == "__main__":
    run_demo()
