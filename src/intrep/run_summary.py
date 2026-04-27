from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from intrep.gpt_model import GPTConfig, gpt_config_to_dict
from intrep.gpt_training import GPTTrainingConfig
from intrep.language_modeling_metrics import language_modeling_metrics_from_training_result


RUN_SUMMARY_SCHEMA = "intrep.run_summary.v1"
RUN_COLLECTION_SCHEMA = "intrep.run_collection.v1"


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def training_config_to_dict(config: GPTTrainingConfig | dict[str, object] | None) -> dict[str, object] | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    return {
        "context_length": config.context_length,
        "batch_size": config.batch_size,
        "batch_stride": config.batch_stride,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }


def model_config_to_dict(config: GPTConfig | dict[str, object] | None) -> dict[str, object] | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    return gpt_config_to_dict(config)


def build_run_summary(
    *,
    kind: str,
    run_id: str | None = None,
    corpus: dict[str, object] | None = None,
    training_config: GPTTrainingConfig | dict[str, object] | None = None,
    model_config: GPTConfig | dict[str, object] | None = None,
    training_loss: dict[str, object] | None = None,
    language_modeling: dict[str, object] | None = None,
    next_observation: dict[str, object] | None = None,
    elapsed_seconds: float | None = None,
    source_path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUN_SUMMARY_SCHEMA,
        "run_id": run_id or new_run_id(),
        "kind": kind,
        "elapsed_seconds": elapsed_seconds,
        "corpus": corpus or {},
        "config": {
            "training": training_config_to_dict(training_config),
            "model": model_config_to_dict(model_config),
        },
        "metrics": {
            "training_loss": training_loss or {},
            "language_modeling": language_modeling or {},
            "next_observation": next_observation or {},
        },
    }
    if source_path is not None:
        payload["source"] = {"path": source_path}
    return payload


def normalize_existing_json(
    payload: dict[str, object],
    *,
    source_path: str | None = None,
) -> dict[str, object]:
    if payload.get("schema_version") == RUN_SUMMARY_SCHEMA:
        return dict(payload)
    if "training_loss" in payload and "language_modeling" in payload:
        return build_run_summary(
            kind="current_experiment",
            corpus=_dict_value(payload.get("corpus")),
            training_config=_dict_value(payload.get("training_config")),
            model_config=_dict_value(payload.get("model_config")),
            training_loss=_dict_value(payload.get("training_loss")),
            language_modeling=_dict_value(payload.get("language_modeling")),
            next_observation=_dict_value(payload.get("next_observation")),
            source_path=source_path,
        )
    if "loss_history" in payload and "initial_train_loss" in payload:
        language_modeling = language_modeling_metrics_from_training_result(
            SimpleNamespace(**payload)
        )
        return build_run_summary(
            kind="train_gpt_loss_history",
            training_config={
                "batch_stride": payload.get("batch_stride"),
                "steps": payload.get("steps"),
            },
            training_loss=dict(payload),
            language_modeling=language_modeling,
            source_path=source_path,
        )
    if "ranking" in payload and "training" in payload:
        return build_run_summary(
            kind="next_observation_evaluation",
            training_config=_dict_value(payload.get("training")),
            next_observation=dict(payload),
            source_path=source_path,
        )
    raise ValueError("unsupported JSON summary shape")


def aggregate_json_outputs(paths: Sequence[str | Path]) -> dict[str, object]:
    runs = []
    for path in paths:
        source_path = str(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"run summary must be a JSON object: {source_path}")
        runs.append(normalize_existing_json(payload, source_path=source_path))
    return {
        "schema_version": RUN_COLLECTION_SCHEMA,
        "run_count": len(runs),
        "runs": runs,
    }


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize or aggregate intrep run JSON outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--input", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "normalize":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("input must be a JSON object")
            output = normalize_existing_json(payload, source_path=str(args.input))
        elif args.command == "aggregate":
            output = aggregate_json_outputs(args.input)
        else:
            raise AssertionError(args.command)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _dict_value(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise ValueError("expected object value")


if __name__ == "__main__":
    main()
