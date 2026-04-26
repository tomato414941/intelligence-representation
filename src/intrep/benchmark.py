from __future__ import annotations

from dataclasses import dataclass

from intrep.evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
from intrep.predictors import FrequencyTransitionPredictor, RuleBasedPredictor
from intrep.transition_data import generate_examples, split_examples
from intrep.update_loop import PredictionErrorUpdateResult, PredictionErrorUpdateLoop, unseen_wallet_case


@dataclass(frozen=True)
class BenchmarkResult:
    train_size: int
    test_size: int
    rule_summary: PredictionEvaluationSummary
    frequency_summary: PredictionEvaluationSummary
    update_result: PredictionErrorUpdateResult

    @property
    def rule_accuracy(self) -> float:
        return self.rule_summary.accuracy

    @property
    def frequency_accuracy(self) -> float:
        return self.frequency_summary.accuracy

    @property
    def update_success(self) -> bool:
        return not self.update_result.before_correct and self.update_result.after_correct


def run_benchmark() -> BenchmarkResult:
    train, test = split_examples(generate_examples())
    test_cases = [example.to_prediction_case() for example in test]

    frequency = FrequencyTransitionPredictor()
    frequency.fit(train)

    update_loop = PredictionErrorUpdateLoop(train)
    update_result = update_loop.update_from_error(unseen_wallet_case())

    return BenchmarkResult(
        train_size=len(train),
        test_size=len(test),
        rule_summary=evaluate_prediction_cases(test_cases, RuleBasedPredictor()),
        frequency_summary=evaluate_prediction_cases(test_cases, frequency),
        update_result=update_result,
    )
