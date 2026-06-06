from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from time import perf_counter

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
)
from intrep.representation.outputs.shogi_legal_move_encoding import shogi_legal_move_feature_ids
from intrep.representation.outputs.shogi_action_plane_policy_encoding import shogi_action_plane_policy_action_index
from intrep.representation.inputs.shogi_position_features.position_features import stack_shogi_position_features


PositionEvaluation = tuple[dict[str, float], float]
PositionEvaluationRequest = tuple[str, tuple[str, ...]]


class ShogiPolicyValueCheckpointEvaluator:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        precision: str = "fp32",
        compile_model: bool = False,
    ) -> ShogiPolicyValueCheckpointEvaluator:
        model = load_shogi_policy_value_checkpoint(checkpoint_path, device=device)
        config = load_shogi_policy_value_checkpoint_training_config(checkpoint_path, device=device)
        torch_device = torch.device(device)
        _validate_inference_precision(precision)
        if compile_model:
            model = torch.compile(model)
        if (
            shogi_policy_value_output_space_for_assembly_spec(config.assembly_spec_id)
            == SHOGI_POLICY_VALUE_OUTPUT_SPACE_ACTION_PLANE_POLICY
        ):
            return cls(_action_plane_policy_evaluator(model, torch_device, config.assembly_spec_id, precision=precision))
        return cls(_legal_move_evaluator(model, torch_device, config.assembly_spec_id, precision=precision))

    def __init__(
        self,
        evaluate_positions: Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]],
    ) -> None:
        self.evaluate_positions = evaluate_positions
        self.last_performance: dict[str, float] = {}

    def evaluate_batch(self, requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        evaluations = self.evaluate_positions(requests)
        self.last_performance = dict(getattr(self.evaluate_positions, "last_performance", {}))
        return evaluations


def _legal_move_evaluator(
    model: torch.nn.Module,
    torch_device: torch.device,
    assembly_spec_id: str,
    *,
    precision: str,
):
    position_features_from_sfen = shogi_position_feature_builder_for_assembly_spec_id(assembly_spec_id)

    def evaluate_batch(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        total_started_at = perf_counter()
        if not requests:
            evaluate_batch.last_performance = _empty_evaluation_performance()
            return []
        phase_started_at = perf_counter()
        boards = [shogi.Board(position_sfen) for position_sfen, _legal_moves in requests]
        board_parse_sec = perf_counter() - phase_started_at
        max_legal_move_count = max(len(legal_moves) for _position_sfen, legal_moves in requests)
        phase_started_at = perf_counter()
        position_feature_rows = [position_features_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
        position_feature_build_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        position_features = stack_shogi_position_features(position_feature_rows)
        position_feature_stack_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        position_features = position_features.to(torch_device)
        _synchronize_torch_device(torch_device)
        position_feature_to_device_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        legal_move_feature_ids = torch.stack(
            [
                shogi_legal_move_feature_ids(
                    legal_moves,
                    turn=board.turn,
                    max_legal_move_count=max_legal_move_count,
                )
                for board, (_position_sfen, legal_moves) in zip(boards, requests, strict=True)
            ]
        )
        legal_move_mask = torch.zeros((len(requests), max_legal_move_count), dtype=torch.bool, device=torch_device)
        for index, (_position_sfen, legal_moves) in enumerate(requests):
            legal_move_mask[index, : len(legal_moves)] = True
        output_feature_build_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        legal_move_feature_ids = legal_move_feature_ids.to(torch_device)
        _synchronize_torch_device(torch_device)
        output_feature_to_device_sec = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        _synchronize_torch_device(torch_device)
        with torch.inference_mode(), _autocast_context(torch_device, precision):
            if hasattr(model, "forward_policy_value"):
                logits, values = model.forward_policy_value(position_features, legal_move_feature_ids, legal_move_mask)
            else:
                logits = model(position_features, legal_move_feature_ids, legal_move_mask)
                values = model.predict_value(position_features) if hasattr(model, "predict_value") else None
        _synchronize_torch_device(torch_device)
        model_forward_sec = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        evaluations: list[PositionEvaluation] = []
        for index, (_position_sfen, legal_moves) in enumerate(requests):
            move_logits = logits[index, : len(legal_moves)]
            evaluations.append((_move_priors(move_logits, legal_moves), _value(values, index)))
        output_decode_sec = perf_counter() - phase_started_at
        evaluate_batch.last_performance = {
            "request_count": float(len(requests)),
            "total_wall_time_sec": perf_counter() - total_started_at,
            "board_parse_sec": board_parse_sec,
            "turn_decode_sec": 0.0,
            "position_feature_build_sec": position_feature_build_sec,
            "position_feature_stack_sec": position_feature_stack_sec,
            "position_feature_to_device_sec": position_feature_to_device_sec,
            "output_feature_build_sec": output_feature_build_sec,
            "output_feature_to_device_sec": output_feature_to_device_sec,
            "model_forward_sec": model_forward_sec,
            "output_decode_sec": output_decode_sec,
        }
        return evaluations

    evaluate_batch.last_performance = _empty_evaluation_performance()
    return evaluate_batch


def _action_plane_policy_evaluator(
    model: torch.nn.Module,
    torch_device: torch.device,
    assembly_spec_id: str,
    *,
    precision: str,
):
    position_features_from_sfen = shogi_position_feature_builder_for_assembly_spec_id(assembly_spec_id)

    def evaluate_batch(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        total_started_at = perf_counter()
        if not requests:
            evaluate_batch.last_performance = _empty_evaluation_performance()
            return []
        phase_started_at = perf_counter()
        turns = [_turn_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
        turn_decode_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        position_feature_rows = [position_features_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
        position_feature_build_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        position_features = stack_shogi_position_features(position_feature_rows)
        position_feature_stack_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        position_features = position_features.to(torch_device)
        _synchronize_torch_device(torch_device)
        position_feature_to_device_sec = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        _synchronize_torch_device(torch_device)
        with torch.inference_mode(), _autocast_context(torch_device, precision):
            if hasattr(model, "forward_policy_value"):
                logits, values = model.forward_policy_value(position_features)
            else:
                logits = model(position_features)
                values = model.predict_value(position_features) if hasattr(model, "predict_value") else None
        _synchronize_torch_device(torch_device)
        model_forward_sec = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        flat_action_indices, flat_batch_indices, offsets = _flat_legal_action_indices(requests, turns)
        action_indices = torch.tensor(flat_action_indices, dtype=torch.long, device=torch_device)
        batch_indices = torch.tensor(flat_batch_indices, dtype=torch.long, device=torch_device)
        output_feature_build_sec = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        flat_move_logits = logits[batch_indices, action_indices]
        evaluations = []
        for index, (_position_sfen, legal_moves) in enumerate(requests):
            start = offsets[index]
            end = offsets[index + 1]
            evaluations.append((_move_priors(flat_move_logits[start:end], legal_moves), _value(values, index)))
        output_decode_sec = perf_counter() - phase_started_at
        evaluate_batch.last_performance = {
            "request_count": float(len(requests)),
            "total_wall_time_sec": perf_counter() - total_started_at,
            "board_parse_sec": 0.0,
            "turn_decode_sec": turn_decode_sec,
            "position_feature_build_sec": position_feature_build_sec,
            "position_feature_stack_sec": position_feature_stack_sec,
            "position_feature_to_device_sec": position_feature_to_device_sec,
            "output_feature_build_sec": output_feature_build_sec,
            "output_feature_to_device_sec": 0.0,
            "model_forward_sec": model_forward_sec,
            "output_decode_sec": output_decode_sec,
        }
        return evaluations

    evaluate_batch.last_performance = _empty_evaluation_performance()
    return evaluate_batch


def _turn_from_sfen(position_sfen: str) -> int:
    parts = position_sfen.split()
    if len(parts) < 2:
        raise ValueError("shogi SFEN must contain side to move")
    if parts[1] == "b":
        return shogi.BLACK
    if parts[1] == "w":
        return shogi.WHITE
    raise ValueError(f"unsupported shogi SFEN side to move: {parts[1]}")


def _flat_legal_action_indices(
    requests: Sequence[PositionEvaluationRequest],
    turns: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    action_indices: list[int] = []
    batch_indices: list[int] = []
    offsets = [0]
    for batch_index, (turn, (_position_sfen, legal_moves)) in enumerate(zip(turns, requests, strict=True)):
        action_indices.extend(_cached_action_plane_policy_action_index(move, turn) for move in legal_moves)
        batch_indices.extend([batch_index] * len(legal_moves))
        offsets.append(len(action_indices))
    return action_indices, batch_indices, offsets


@lru_cache(maxsize=65536)
def _cached_action_plane_policy_action_index(move_usi: str, turn: int) -> int:
    return shogi_action_plane_policy_action_index(move_usi, turn=turn)


def _validate_inference_precision(precision: str) -> None:
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported shogi checkpoint inference precision: {precision}")


def _autocast_context(torch_device: torch.device, precision: str):
    if precision == "bf16" and torch_device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _empty_evaluation_performance() -> dict[str, float]:
    return {
        "request_count": 0.0,
        "total_wall_time_sec": 0.0,
        "board_parse_sec": 0.0,
        "turn_decode_sec": 0.0,
        "position_feature_build_sec": 0.0,
        "position_feature_stack_sec": 0.0,
        "position_feature_to_device_sec": 0.0,
        "output_feature_build_sec": 0.0,
        "output_feature_to_device_sec": 0.0,
        "model_forward_sec": 0.0,
        "output_decode_sec": 0.0,
    }


def _synchronize_torch_device(torch_device: torch.device) -> None:
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _move_priors(move_logits: torch.Tensor, legal_moves: tuple[str, ...]) -> dict[str, float]:
    probabilities = torch.softmax(move_logits, dim=0).detach().cpu().tolist()
    return {move: float(probabilities[move_index]) for move_index, move in enumerate(legal_moves)}


def _value(values: torch.Tensor | None, index: int) -> float:
    if values is None:
        return 0.0
    return float(values[index].detach().cpu().item())
