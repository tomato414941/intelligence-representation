from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import time

from intrep.generated_environment_corpus import EVAL_SLICES, generated_environment_corpus_selection
from intrep.gpt_model import GPTConfig
from intrep.gpt_training import GPTTrainingConfig
from intrep.run_summary import build_run_summary, compare_json_outputs, write_json


PipelineRunner = Callable[..., dict[str, object]]


EXPERIMENT_FAILURE_REPORT_SCHEMA = "intrep.experiment_failure_report.v1"


@dataclass(frozen=True)
class SweepRun:
    seed: int
    eval_slice: str
    run_id: str
    summary_path: Path


@dataclass(frozen=True)
class SweepFailure:
    seed: int
    eval_slice: str
    error_type: str
    message: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    reason: str = "exception"


@dataclass(frozen=True)
class SweepResult:
    output_dir: Path
    runs: list[SweepRun]
    failures: list[SweepFailure]
    comparison_path: Path
    failure_report_path: Path


def run_generated_environment_sweep(
    *,
    output_dir: str | Path,
    seeds: Sequence[int],
    eval_slices: Sequence[str] = EVAL_SLICES,
    context_length: int = 64,
    batch_size: int = 8,
    batch_stride: int | None = None,
    max_steps: int = 20,
    learning_rate: float = 0.003,
    device: str = "cpu",
    distractor_policy: str = "same_entity",
    model_config: GPTConfig | None = None,
    runner: PipelineRunner | None = None,
) -> SweepResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_summary_paths: list[Path] = []
    runs: list[SweepRun] = []
    failures: list[SweepFailure] = []
    experiment_runner = runner or _run_current_experiment

    for seed in seeds:
        for eval_slice in eval_slices:
            try:
                selection = generated_environment_corpus_selection(eval_slice)
                training_config = GPTTrainingConfig(
                    context_length=context_length,
                    batch_size=batch_size,
                    batch_stride=batch_stride,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    seed=seed,
                    device=device,  # type: ignore[arg-type]
                )
                started_at = time.perf_counter()
                summary = _call_runner(
                    experiment_runner,
                    selection.train_documents,
                    eval_documents=selection.eval_documents,
                    corpus_label="generated-environment",
                    eval_corpus_label=selection.eval_label,
                    training_config=training_config,
                    model_config=model_config,
                    distractor_policy=distractor_policy,
                )
                elapsed_seconds = time.perf_counter() - started_at
                run_id = f"generated-environment-seed-{seed}-{eval_slice}"
                summary_path = output_root / f"{run_id}.json"
                write_json(
                    summary_path,
                    build_run_summary(
                        kind="current_experiment",
                        run_id=run_id,
                        corpus={
                            "train": {"label": "generated-environment"},
                            "eval": {"label": selection.eval_label},
                            "eval_slice": selection.eval_label,
                        },
                        training_config=summary.get("training_config", training_config),  # type: ignore[arg-type]
                        model_config=summary.get("model_config"),  # type: ignore[arg-type]
                        training_loss=summary.get("training_loss"),  # type: ignore[arg-type]
                        language_modeling=summary.get("language_modeling"),  # type: ignore[arg-type]
                        next_observation=summary.get("next_observation"),  # type: ignore[arg-type]
                        symbolic_to_natural=summary.get("symbolic_to_natural"),  # type: ignore[arg-type]
                        elapsed_seconds=elapsed_seconds,
                    ),
                )
                run_summary_paths.append(summary_path)
                failures.extend(_metric_failures(summary, seed=seed, eval_slice=eval_slice))
                runs.append(
                    SweepRun(
                        seed=seed,
                        eval_slice=eval_slice,
                        run_id=run_id,
                        summary_path=summary_path,
                    )
                )
            except Exception as error:  # noqa: BLE001 - report all failed sweep cells.
                failures.append(
                    SweepFailure(
                        seed=seed,
                        eval_slice=eval_slice,
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )

    comparison_path = output_root / "comparison.json"
    failure_report_path = output_root / "failures.json"
    comparison = compare_json_outputs(run_summary_paths) if run_summary_paths else _empty_comparison()
    write_json(comparison_path, comparison)
    write_json(
        failure_report_path,
        _failure_report(failures, run_count=len(runs) + len([failure for failure in failures if failure.reason == "exception"])),
    )
    return SweepResult(
        output_dir=output_root,
        runs=runs,
        failures=failures,
        comparison_path=comparison_path,
        failure_report_path=failure_report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generated-environment seed x slice experiment sweeps.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--eval-slice", choices=EVAL_SLICES, action="append")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--distractor-policy",
        choices=("all_other", "hard", "same_entity"),
        default="same_entity",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_generated_environment_sweep(
        output_dir=args.output_dir,
        seeds=args.seed,
        eval_slices=args.eval_slice or EVAL_SLICES,
        context_length=args.context_length,
        batch_size=args.batch_size,
        batch_stride=args.batch_stride,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        distractor_policy=args.distractor_policy,
    )
    print(
        f"wrote runs={len(result.runs)} failures={len(result.failures)} "
        f"comparison={result.comparison_path} failures={result.failure_report_path}"
    )


def _run_current_experiment(*args: object, **kwargs: object) -> dict[str, object]:
    from intrep.current_experiment import run_current_experiment

    return run_current_experiment(*args, **kwargs)


def _call_runner(
    runner: PipelineRunner,
    documents: object,
    **kwargs: object,
) -> dict[str, object]:
    signature = inspect.signature(runner)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return runner(documents, **kwargs)
    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return runner(documents, **accepted_kwargs)


def _empty_comparison() -> dict[str, object]:
    return {
        "schema_version": "intrep.run_comparison.v1",
        "run_count": 0,
        "metrics": [],
        "runs": [],
    }


def _failure_report(failures: Sequence[SweepFailure], *, run_count: int) -> dict[str, object]:
    return {
        "schema_version": EXPERIMENT_FAILURE_REPORT_SCHEMA,
        "run_count": run_count,
        "failure_count": len(failures),
        "failures": [
            {
                "seed": failure.seed,
                "eval_slice": failure.eval_slice,
                "error_type": failure.error_type,
                "message": failure.message,
                "metric": failure.metric,
                "value": failure.value,
                "threshold": failure.threshold,
                "reason": failure.reason,
            }
            for failure in failures
        ],
    }


def _metric_failures(
    summary: dict[str, object],
    *,
    seed: int,
    eval_slice: str,
) -> list[SweepFailure]:
    failures: list[SweepFailure] = []
    for metric_path in (
        "language_modeling.initial_eval_loss",
        "language_modeling.final_eval_loss",
        "next_observation.before.top1_accuracy",
        "next_observation.after.top1_accuracy",
        "symbolic_to_natural.before.top1_accuracy",
        "symbolic_to_natural.after.top1_accuracy",
    ):
        if _value_at_path(summary, metric_path) is None:
            failures.append(
                _metric_failure(
                    seed=seed,
                    eval_slice=eval_slice,
                    metric=metric_path,
                    value=None,
                    threshold=None,
                    reason="missing_metric",
                )
            )
    initial_eval_loss = _float_at_path(summary, "language_modeling.initial_eval_loss")
    final_eval_loss = _float_at_path(summary, "language_modeling.final_eval_loss")
    if initial_eval_loss is not None and final_eval_loss is not None and final_eval_loss > initial_eval_loss:
        failures.append(
            _metric_failure(
                seed=seed,
                eval_slice=eval_slice,
                metric="language_modeling.final_eval_loss",
                value=final_eval_loss,
                threshold=initial_eval_loss,
                reason="regression",
            )
        )
    for prefix in ("next_observation", "symbolic_to_natural"):
        before = _float_at_path(summary, f"{prefix}.before.top1_accuracy")
        after = _float_at_path(summary, f"{prefix}.after.top1_accuracy")
        if before is not None and after is not None and after < before:
            failures.append(
                _metric_failure(
                    seed=seed,
                    eval_slice=eval_slice,
                    metric=f"{prefix}.after.top1_accuracy",
                    value=after,
                    threshold=before,
                    reason="regression",
                )
            )
    return failures


def _metric_failure(
    *,
    seed: int,
    eval_slice: str,
    metric: str,
    value: float | None,
    threshold: float | None,
    reason: str,
) -> SweepFailure:
    return SweepFailure(
        seed=seed,
        eval_slice=eval_slice,
        error_type="MetricFailure",
        message=f"{metric} {reason}",
        metric=metric,
        value=value,
        threshold=threshold,
        reason=reason,
    )


def _float_at_path(payload: dict[str, object], path: str) -> float | None:
    value = _value_at_path(payload, path)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _value_at_path(payload: dict[str, object], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


if __name__ == "__main__":
    main()
