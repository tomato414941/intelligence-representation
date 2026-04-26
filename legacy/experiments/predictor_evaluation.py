from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

try:
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.predictor_interface import Action, Fact, Predictor, RuleBasedPredictor


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


class PredictorFactory(Protocol):
    def __call__(self) -> Predictor:
        ...


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


def smoke_cases() -> list[PredictionCase]:
    return [
        PredictionCase(
            name="place_book_library",
            initial_state=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="place", actor="佐藤", object="本", target="図書館"),
            expected_fact=Fact(subject="本", predicate="located_at", object="図書館"),
        ),
        PredictionCase(
            name="unsupported_throw",
            initial_state=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="throw", actor="佐藤", object="本", target="床"),
            expected_fact=None,
        ),
    ]


def run_demo() -> None:
    summary = evaluate_prediction_cases(smoke_cases(), RuleBasedPredictor())
    for result in summary.results:
        predicted = result.predicted_fact.render() if result.predicted_fact else "unsupported"
        expected = result.expected_fact.render() if result.expected_fact else "unsupported"
        print(f"{result.case_name}: predicted={predicted} expected={expected} correct={result.correct}")
    print(f"accuracy={summary.accuracy:.2f} unsupported_rate={summary.unsupported_rate:.2f}")


if __name__ == "__main__":
    run_demo()
