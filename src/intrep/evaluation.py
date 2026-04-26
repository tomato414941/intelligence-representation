from __future__ import annotations

from dataclasses import dataclass

from intrep.types import Action, Fact, Predictor


@dataclass(frozen=True)
class PredictionCase:
    name: str
    initial_state: list[Fact]
    action: Action
    expected_fact: Fact | None


@dataclass(frozen=True)
class PredictionCaseResult:
    case_name: str
    predicted_fact: Fact | None
    expected_fact: Fact | None
    correct: bool
    unsupported: bool


@dataclass(frozen=True)
class PredictionEvaluationSummary:
    results: list[PredictionCaseResult]
    accuracy: float
    unsupported_rate: float


def evaluate_prediction_case(case: PredictionCase, predictor: Predictor) -> PredictionCaseResult:
    predicted = predictor.predict(case.initial_state, case.action)
    unsupported = predicted is None
    if case.expected_fact is None:
        correct = predicted is None
    elif predicted is None:
        correct = False
    else:
        correct = predicted.key() == case.expected_fact.key()

    return PredictionCaseResult(
        case_name=case.name,
        predicted_fact=predicted,
        expected_fact=case.expected_fact,
        correct=correct,
        unsupported=unsupported,
    )


def evaluate_prediction_cases(cases: list[PredictionCase], predictor: Predictor) -> PredictionEvaluationSummary:
    results = [evaluate_prediction_case(case, predictor) for case in cases]
    if not results:
        return PredictionEvaluationSummary(results=[], accuracy=0.0, unsupported_rate=0.0)

    return PredictionEvaluationSummary(
        results=results,
        accuracy=sum(1 for result in results if result.correct) / len(results),
        unsupported_rate=sum(1 for result in results if result.unsupported) / len(results),
    )

