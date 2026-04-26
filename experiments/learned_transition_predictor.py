from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intrep.environment import MiniTransitionEnvironment
from intrep.predictors import FrequencyTransitionPredictor
from intrep.transition_data import (
    LearnedTransitionComparison,
    compare_predictors,
    generate_examples,
    smoke_comparison,
    split_examples,
)


def run_demo() -> None:
    comparison = smoke_comparison()
    print(f"train_size={comparison.train_size} test_size={comparison.test_size}")
    print(
        f"rule_accuracy={comparison.rule_summary.accuracy:.2f} "
        f"rule_unsupported={comparison.rule_summary.unsupported_rate:.2f}"
    )
    print(
        f"learned_accuracy={comparison.learned_summary.accuracy:.2f} "
        f"learned_unsupported={comparison.learned_summary.unsupported_rate:.2f}"
    )


if __name__ == "__main__":
    run_demo()
