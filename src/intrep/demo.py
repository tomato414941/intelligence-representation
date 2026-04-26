from __future__ import annotations

from intrep.benchmark import run_benchmark


def main() -> None:
    result = run_benchmark()

    print("intrep prototype demo")
    print(
        "benchmark:"
        f" train={result.train_size}"
        f" test={len(result.frequency_summary.results)}"
        f" rule_accuracy={result.rule_accuracy:.2f}"
        f" frequency_accuracy={result.frequency_accuracy:.2f}"
        f" state_aware_accuracy={result.state_aware_accuracy:.2f}"
    )
    for benchmark_slice in result.slices:
        print(
            "slice:"
            f" name={benchmark_slice.name}"
            f" cases={benchmark_slice.case_count}"
            f" frequency_accuracy={benchmark_slice.frequency_summary.accuracy:.2f}"
            f" frequency_unsupported={benchmark_slice.frequency_summary.unsupported_rate:.2f}"
            f" state_aware_accuracy={benchmark_slice.state_aware_summary.accuracy:.2f}"
        )
    print(
        "prediction_error_update:"
        f" case={result.update_result.case_name}"
        f" error={result.update_result.prediction_error_type}"
        f" before_correct={result.update_result.before_correct}"
        f" after_correct={result.update_result.after_correct}"
        f" training={result.update_result.training_size_before}->{result.update_result.training_size_after}"
    )


if __name__ == "__main__":
    main()
