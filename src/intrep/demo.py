from __future__ import annotations

from intrep.transition_data import smoke_comparison
from intrep.update_loop import smoke_update_result


def main() -> None:
    comparison = smoke_comparison()
    update = smoke_update_result()

    print("intrep prototype demo")
    print(
        "learned_transition:"
        f" train={comparison.train_size}"
        f" test={comparison.test_size}"
        f" rule_accuracy={comparison.rule_summary.accuracy:.2f}"
        f" learned_accuracy={comparison.learned_summary.accuracy:.2f}"
    )
    print(
        "prediction_error_update:"
        f" case={update.case_name}"
        f" error={update.prediction_error_type}"
        f" before_correct={update.before_correct}"
        f" after_correct={update.after_correct}"
        f" training={update.training_size_before}->{update.training_size_after}"
    )


if __name__ == "__main__":
    main()
