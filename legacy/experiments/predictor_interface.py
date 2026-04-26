from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


PredictionStatus = Literal["pending", "confirmed", "mismatch", "unsupported"]
ErrorType = Literal["none", "mismatch", "unsupported"]


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
    expected: Fact | None
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
    expected: Fact | None
    observed: Fact | None


class Predictor(Protocol):
    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        ...


class RuleBasedPredictor:
    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if action.type == "place":
            return Fact(subject=action.object, predicate="located_at", object=action.target)
        return None


class PredictiveWorld:
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.state: list[Fact] = []
        self.predictions: list[Prediction] = []
        self.observations: list[Observation] = []
        self.errors: list[PredictionError] = []
        self._next_prediction_id = 1
        self._next_observation_id = 1
        self._next_error_id = 1

    def add_fact(self, fact: Fact) -> None:
        if fact.key() not in {existing.key() for existing in self.state}:
            self.state.append(fact)

    def predict(self, action: Action) -> Prediction:
        expected = self.predictor.predict(self.state, action)
        status: PredictionStatus = "pending" if expected else "unsupported"
        prediction = Prediction(
            id=f"prediction_{self._next_prediction_id}",
            action=action,
            expected=expected,
            status=status,
        )
        self._next_prediction_id += 1
        self.predictions.append(prediction)
        return prediction

    def observe(self, fact: Fact) -> Observation:
        observation = Observation(id=f"obs_{self._next_observation_id}", fact=fact)
        self._next_observation_id += 1
        self.observations.append(observation)
        self.add_fact(fact)
        return observation

    def compare(self, prediction: Prediction, observation: Observation | None) -> PredictionError:
        observed = observation.fact if observation else None
        if prediction.expected is None:
            prediction.status = "unsupported"
            error_type: ErrorType = "unsupported"
        elif observed is not None and prediction.expected.key() == observed.key():
            prediction.status = "confirmed"
            error_type = "none"
        else:
            prediction.status = "mismatch"
            error_type = "mismatch"

        error = PredictionError(
            id=f"error_{self._next_error_id}",
            prediction=prediction.id,
            type=error_type,
            expected=prediction.expected,
            observed=observed,
        )
        self._next_error_id += 1
        self.errors.append(error)
        return error


def run_demo() -> None:
    world = PredictiveWorld(RuleBasedPredictor())
    prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
    observation = world.observe(Fact(subject="本", predicate="located_at", object="図書館"))
    error = world.compare(prediction, observation)
    expected = prediction.expected.render() if prediction.expected else "none"
    print(f"prediction={expected} status={prediction.status} error={error.type}")


if __name__ == "__main__":
    run_demo()
