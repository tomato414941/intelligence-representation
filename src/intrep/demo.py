from __future__ import annotations

from intrep.benchmark import run_benchmark


def main() -> None:
    result = run_benchmark()

    print("intrep prototype demo")
    print(
        "benchmark:"
        f" train={result.train_size}"
        f" test={result.test_size}"
        f" rule_accuracy={result.rule_accuracy:.2f}"
        f" frequency_accuracy={result.frequency_accuracy:.2f}"
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
