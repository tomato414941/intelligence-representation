from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intrep.dataset import ActionConditionedExample
from intrep.predictors import FrequencyTransitionPredictor
from intrep.transition_data import generate_examples, held_out_object_examples, split_examples
from intrep.types import Action, Fact


PredictionErrorType = Literal["none", "mismatch", "unsupported"]


@dataclass(frozen=True)
class PredictionErrorUpdateResult:
    case_name: str
    prediction_error_type: PredictionErrorType
    predicted_before: Fact | None
    predicted_after: Fact | None
    observed: Fact
    before_correct: bool
    after_correct: bool
    training_size_before: int
    training_size_after: int


class PredictionErrorUpdateLoop:
    def __init__(self, training_examples: list[ActionConditionedExample]) -> None:
        self.training_examples = list(training_examples)
        self.predictor = FrequencyTransitionPredictor()
        self.predictor.fit(self.training_examples)

    def update_from_error(self, case: ActionConditionedExample) -> PredictionErrorUpdateResult:
        if case.expected_observation is None:
            raise ValueError("Prediction error update requires an observed fact.")

        training_size_before = len(self.training_examples)
        predicted_before = self.predictor.predict(case.state_before, case.action)
        error_type = _prediction_error_type(predicted_before, case.expected_observation)

        if error_type != "none":
            self.training_examples.append(case)
            self.predictor.fit(self.training_examples)

        predicted_after = self.predictor.predict(case.state_before, case.action)
        return PredictionErrorUpdateResult(
            case_name=case.id,
            prediction_error_type=error_type,
            predicted_before=predicted_before,
            predicted_after=predicted_after,
            observed=case.expected_observation,
            before_correct=_same_fact(predicted_before, case.expected_observation),
            after_correct=_same_fact(predicted_after, case.expected_observation),
            training_size_before=training_size_before,
            training_size_after=len(self.training_examples),
        )


def unseen_wallet_case() -> ActionConditionedExample:
    return held_out_object_examples()[0]


def smoke_update_result() -> PredictionErrorUpdateResult:
    train, _ = split_examples(generate_examples())
    loop = PredictionErrorUpdateLoop(train)
    return loop.update_from_error(unseen_wallet_case())


def _prediction_error_type(predicted: Fact | None, observed: Fact) -> PredictionErrorType:
    if predicted is None:
        return "unsupported"
    if predicted.key() != observed.key():
        return "mismatch"
    return "none"


def _same_fact(left: Fact | None, right: Fact) -> bool:
    return left is not None and left.key() == right.key()
