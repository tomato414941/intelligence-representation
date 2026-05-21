from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Sequence

import shogi
import torch

from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint,
    load_shogi_policy_value_checkpoint_training_config,
)
from intrep.problems.shogi_policy_value.output_space import (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_ACTION_PLANE_POLICY,
    shogi_policy_value_output_space_for_assembly_spec,
)
from intrep.problems.shogi_policy_value.position_input_identity import (
    shogi_position_feature_builder_for_assembly_spec_id,
    shogi_position_input_identity_for_assembly_spec_id,
)
from intrep.representation.outputs.shogi_legal_move_encoding import shogi_legal_move_feature_ids
from intrep.representation.outputs.shogi_action_plane_policy_encoding import shogi_action_plane_policy_legal_mask
from intrep.representation.inputs.shogi_position_features.position_rich import (
    SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_RICH_POSITION_INPUT_SCHEMA_ID,
    shogi_rich_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_features import stack_shogi_position_features


SHOGI_POSITION_FEATURE_GENERATION_BENCHMARK_SCHEMA = (
    "intrep.problems.shogi_policy_value.position_feature_generation_benchmark.v1"
)
SHOGI_POLICY_VALUE_INFERENCE_BATCHING_BENCHMARK_SCHEMA = (
    "intrep.problems.shogi_policy_value.inference_batching_benchmark.v1"
)


def load_position_sfens_from_jsonl(
    path: str | Path,
    *,
    sfen_field: str = "position_sfen",
    limit: int | None = None,
) -> list[str]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    position_sfens: list[str] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if limit is not None and len(position_sfens) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            position_sfen = payload.get(sfen_field)
            if not isinstance(position_sfen, str) or not position_sfen:
                raise ValueError(f"line {line_number} missing non-empty {sfen_field!r}")
            position_sfens.append(position_sfen)
    if not position_sfens:
        raise ValueError("positions JSONL did not contain any positions")
    return position_sfens


def latency_summary_ms(durations_seconds: Sequence[float]) -> dict[str, float]:
    if not durations_seconds:
        raise ValueError("durations_seconds must not be empty")
    values = sorted(duration * 1000.0 for duration in durations_seconds)
    return {
        "min": values[0],
        "mean": sum(values) / len(values),
        "median": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": values[-1],
    }


def benchmark_shogi_position_feature_generation(
    position_sfens: Sequence[str],
    *,
    warmup: int = 1,
    repeat: int = 1,
) -> dict[str, object]:
    if not position_sfens:
        raise ValueError("position_sfens must not be empty")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    positions = tuple(position_sfens)

    for _ in range(warmup):
        for position_sfen in positions:
            shogi_rich_position_features_from_sfen(position_sfen)

    durations: list[float] = []
    wall_started = time.perf_counter()
    for _ in range(repeat):
        for position_sfen in positions:
            started = time.perf_counter()
            shogi_rich_position_features_from_sfen(position_sfen)
            durations.append(time.perf_counter() - started)
    wall_time_seconds = time.perf_counter() - wall_started
    measured_position_count = len(durations)
    measured_work_seconds = sum(durations)

    return {
        "schema_version": SHOGI_POSITION_FEATURE_GENERATION_BENCHMARK_SCHEMA,
        "input_schema_id": SHOGI_RICH_POSITION_INPUT_SCHEMA_ID,
        "input_feature_manifest_hash": SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH,
        "position_count": len(positions),
        "measured_position_count": measured_position_count,
        "warmup": warmup,
        "repeat": repeat,
        "latency_ms": latency_summary_ms(durations),
        "positions_per_second": _throughput(measured_position_count, measured_work_seconds),
        "wall_time_seconds": wall_time_seconds,
    }


def benchmark_shogi_policy_value_inference_batching(
    checkpoint_path: str | Path,
    position_sfens: Sequence[str],
    *,
    batch_sizes: Sequence[int],
    device: str = "cpu",
    dtype: str = "float32",
    warmup_batches: int = 1,
    measure_batches: int = 3,
) -> dict[str, object]:
    if not position_sfens:
        raise ValueError("position_sfens must not be empty")
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if measure_batches <= 0:
        raise ValueError("measure_batches must be positive")
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("batch sizes must be positive")

    torch_device = torch.device(device)
    torch_dtype = _torch_dtype(dtype)
    model = load_shogi_policy_value_checkpoint(checkpoint_path, device=device)
    if torch_dtype != torch.float32:
        model = model.to(dtype=torch_dtype)
    model.eval()
    config = load_shogi_policy_value_checkpoint_training_config(checkpoint_path, device=device)
    output_space = shogi_policy_value_output_space_for_assembly_spec(config.assembly_spec_id)
    positions = tuple(position_sfens)

    batch_results: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            for batch_index in range(warmup_batches):
                batch_positions = _cycled_batch(positions, batch_size, batch_index)
                _run_policy_value_inference_batch(
                    model,
                    output_space=output_space,
                    assembly_spec_id=config.assembly_spec_id,
                    position_sfens=batch_positions,
                    device=torch_device,
                )
            _synchronize_if_needed(torch_device)

            durations: list[float] = []
            total_outputs = 0
            wall_started = time.perf_counter()
            for batch_index in range(measure_batches):
                batch_positions = _cycled_batch(positions, batch_size, warmup_batches + batch_index)
                _synchronize_if_needed(torch_device)
                started = time.perf_counter()
                output_count = _run_policy_value_inference_batch(
                    model,
                    output_space=output_space,
                    assembly_spec_id=config.assembly_spec_id,
                    position_sfens=batch_positions,
                    device=torch_device,
                )
                _synchronize_if_needed(torch_device)
                durations.append(time.perf_counter() - started)
                total_outputs += output_count
            wall_time_seconds = time.perf_counter() - wall_started
            measured_position_count = batch_size * measure_batches
            measured_work_seconds = sum(durations)

            batch_results.append(
                {
                    "batch_size": batch_size,
                    "warmup_batches": warmup_batches,
                    "measured_batches": measure_batches,
                    "measured_position_count": measured_position_count,
                    "output_element_count": total_outputs,
                    "latency_ms": latency_summary_ms(durations),
                    "positions_per_second": _throughput(measured_position_count, measured_work_seconds),
                    "wall_time_seconds": wall_time_seconds,
                }
            )

    input_identity = shogi_position_input_identity_for_assembly_spec_id(config.assembly_spec_id)
    return {
        "schema_version": SHOGI_POLICY_VALUE_INFERENCE_BATCHING_BENCHMARK_SCHEMA,
        "input_schema_id": input_identity["input_schema_id"],
        "input_feature_manifest_hash": input_identity["input_feature_manifest_hash"],
        "checkpoint_path": str(checkpoint_path),
        "assembly_spec_id": config.assembly_spec_id,
        "output_space": output_space,
        "device": str(torch_device),
        "dtype": dtype,
        "position_count": len(positions),
        "includes_feature_generation": True,
        "batch_results": batch_results,
    }


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    batch_sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not batch_sizes:
        raise ValueError("batch sizes must not be empty")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    return batch_sizes


def write_json_result(result: dict[str, object], path: str | Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    Path(path).write_text(text, encoding="utf-8")


def _run_policy_value_inference_batch(
    model: torch.nn.Module,
    *,
    output_space: str,
    assembly_spec_id: str,
    position_sfens: Sequence[str],
    device: torch.device,
) -> int:
    boards = [shogi.Board(position_sfen) for position_sfen in position_sfens]
    position_features_from_sfen = shogi_position_feature_builder_for_assembly_spec_id(assembly_spec_id)
    position_features = stack_shogi_position_features(
        [position_features_from_sfen(position_sfen) for position_sfen in position_sfens]
    ).to(device)
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_ACTION_PLANE_POLICY:
        legal_action_mask = torch.stack([shogi_action_plane_policy_legal_mask(board) for board in boards]).to(device)
        logits, values = model.forward_policy_value(position_features, legal_action_mask)
    else:
        legal_moves_by_position = [tuple(move.usi() for move in board.legal_moves) for board in boards]
        max_legal_move_count = max(len(legal_moves) for legal_moves in legal_moves_by_position)
        legal_move_feature_ids = torch.stack(
            [
                shogi_legal_move_feature_ids(
                    legal_moves,
                    turn=board.turn,
                    max_legal_move_count=max_legal_move_count,
                )
                for board, legal_moves in zip(boards, legal_moves_by_position, strict=True)
            ]
        ).to(device)
        legal_move_mask = torch.zeros((len(boards), max_legal_move_count), dtype=torch.bool, device=device)
        for index, legal_moves in enumerate(legal_moves_by_position):
            legal_move_mask[index, : len(legal_moves)] = True
        logits, values = model.forward_policy_value(position_features, legal_move_feature_ids, legal_move_mask)
    return int(logits.numel() + values.numel())


def _cycled_batch(values: Sequence[str], batch_size: int, batch_index: int) -> tuple[str, ...]:
    offset = batch_index * batch_size
    return tuple(values[(offset + index) % len(values)] for index in range(batch_size))


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _throughput(count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return float("inf")
    return count / seconds


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
