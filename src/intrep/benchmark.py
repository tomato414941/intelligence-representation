from __future__ import annotations

from dataclasses import dataclass

from intrep.evaluation import PredictionEvaluationSummary, evaluate_prediction_cases
from intrep.predictors import (
    FrequencyTransitionPredictor,
    RuleBasedPredictor,
    StateAwarePredictor,
)
from intrep.transition_data import (
    generate_examples,
    generated_find_examples,
    held_out_object_examples,
    longer_chain_examples,
    missing_link_examples,
    noisy_distractor_examples,
    split_generated_examples,
    split_examples,
)
from intrep.update_loop import PredictionErrorUpdateResult, PredictionErrorUpdateLoop, unseen_wallet_case


@dataclass(frozen=True)
class BenchmarkSliceResult:
    name: str
    case_count: int
    rule_summary: PredictionEvaluationSummary
    frequency_summary: PredictionEvaluationSummary
    state_aware_summary: PredictionEvaluationSummary


@dataclass(frozen=True)
class BenchmarkResult:
    train_size: int
    test_size: int
    rule_summary: PredictionEvaluationSummary
    frequency_summary: PredictionEvaluationSummary
    state_aware_summary: PredictionEvaluationSummary
    slices: list[BenchmarkSliceResult]
    generated_train_size: int
    generated_slices: list[BenchmarkSliceResult]
    update_result: PredictionErrorUpdateResult

    @property
    def rule_accuracy(self) -> float:
        return self.rule_summary.accuracy

    @property
    def frequency_accuracy(self) -> float:
        return self.frequency_summary.accuracy

    @property
    def state_aware_accuracy(self) -> float:
        return self.state_aware_summary.accuracy

    @property
    def update_success(self) -> bool:
        return not self.update_result.before_correct and self.update_result.after_correct

    def slice(self, name: str) -> BenchmarkSliceResult:
        for result in self.slices:
            if result.name == name:
                return result
        raise KeyError(name)

    def generated_slice(self, name: str) -> BenchmarkSliceResult:
        for result in self.generated_slices:
            if result.name == name:
                return result
        raise KeyError(name)


def run_benchmark() -> BenchmarkResult:
    train, test = split_examples(generate_examples())
    seen_cases = [example.to_prediction_case() for example in test]
    held_out_object_cases = [example.to_prediction_case() for example in held_out_object_examples()]
    longer_chain_cases = [example.to_prediction_case() for example in longer_chain_examples()]
    missing_link_cases = [example.to_prediction_case() for example in missing_link_examples()]
    noisy_distractor_cases = [example.to_prediction_case() for example in noisy_distractor_examples()]
    test_cases = (
        seen_cases
        + held_out_object_cases
        + longer_chain_cases
        + missing_link_cases
        + noisy_distractor_cases
    )

    frequency = FrequencyTransitionPredictor()
    frequency.fit(train)
    state_aware = StateAwarePredictor()
    state_aware.fit(train)
    rule = RuleBasedPredictor()

    update_loop = PredictionErrorUpdateLoop(train)
    update_result = update_loop.update_from_error(unseen_wallet_case())
    generated_train, generated_slice_examples = split_generated_examples(generated_find_examples())
    generated_predictors = _fit_predictors(generated_train)

    return BenchmarkResult(
        train_size=len(train),
        test_size=len(test),
        rule_summary=evaluate_prediction_cases(test_cases, rule),
        frequency_summary=evaluate_prediction_cases(test_cases, frequency),
        state_aware_summary=evaluate_prediction_cases(test_cases, state_aware),
        slices=[
            _evaluate_slice(
                "seen_action_patterns",
                seen_cases,
                rule,
                frequency,
                state_aware,
            ),
            _evaluate_slice(
                "held_out_object",
                held_out_object_cases,
                rule,
                frequency,
                state_aware,
            ),
            _evaluate_slice(
                "longer_chain",
                longer_chain_cases,
                rule,
                frequency,
                state_aware,
            ),
            _evaluate_slice(
                "missing_link",
                missing_link_cases,
                rule,
                frequency,
                state_aware,
            ),
            _evaluate_slice(
                "noisy_distractor",
                noisy_distractor_cases,
                rule,
                frequency,
                state_aware,
            ),
        ],
        generated_train_size=len(generated_train),
        generated_slices=[
            _evaluate_slice(
                name,
                [example.to_prediction_case() for example in examples],
                *generated_predictors,
            )
            for name, examples in generated_slice_examples.items()
        ],
        update_result=update_result,
    )


def _fit_predictors(
    train,
) -> tuple[
    RuleBasedPredictor,
    FrequencyTransitionPredictor,
    StateAwarePredictor,
]:
    frequency = FrequencyTransitionPredictor()
    frequency.fit(train)
    state_aware = StateAwarePredictor()
    state_aware.fit(train)
    return RuleBasedPredictor(), frequency, state_aware


def _evaluate_slice(
    name: str,
    cases,
    rule: RuleBasedPredictor,
    frequency: FrequencyTransitionPredictor,
    state_aware: StateAwarePredictor,
) -> BenchmarkSliceResult:
    return BenchmarkSliceResult(
        name=name,
        case_count=len(cases),
        rule_summary=evaluate_prediction_cases(cases, rule),
        frequency_summary=evaluate_prediction_cases(cases, frequency),
        state_aware_summary=evaluate_prediction_cases(cases, state_aware),
    )
