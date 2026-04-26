from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intrep.update_loop import (
    PredictionErrorType,
    PredictionErrorUpdateLoop,
    PredictionErrorUpdateResult,
    smoke_update_result,
    unseen_wallet_case,
)


def run_demo() -> None:
    result = smoke_update_result()
    before = result.predicted_before.render() if result.predicted_before else "unsupported"
    after = result.predicted_after.render() if result.predicted_after else "unsupported"
    print(
        f"{result.case_name}: error={result.prediction_error_type} "
        f"before={before} after={after} observed={result.observed.render()} "
        f"before_correct={result.before_correct} after_correct={result.after_correct} "
        f"train={result.training_size_before}->{result.training_size_after}"
    )


if __name__ == "__main__":
    run_demo()
