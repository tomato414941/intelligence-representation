from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import shogi
import torch

from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint,
    load_shogi_policy_value_checkpoint_training_config,
)
from intrep.problems.shogi_policy_value.model import SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID
from intrep.worlds.shogi.move_encoding import shogi_legal_move_features
from intrep.worlds.shogi.policy_plane import shogi_policy_plane_action_index, shogi_policy_plane_legal_mask
from intrep.worlds.shogi.position_encoding import shogi_position_features_from_sfen, stack_shogi_position_features


PositionEvaluation = tuple[dict[str, float], float]
PositionEvaluationRequest = tuple[str, tuple[str, ...]]


class ShogiPolicyValueCheckpointEvaluator:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> ShogiPolicyValueCheckpointEvaluator:
        model = load_shogi_policy_value_checkpoint(checkpoint_path, device=device)
        config = load_shogi_policy_value_checkpoint_training_config(checkpoint_path, device=device)
        torch_device = torch.device(device)
        if config.policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
            return cls(_policy_plane_evaluator(model, torch_device))
        return cls(_legal_move_evaluator(model, torch_device))

    def __init__(
        self,
        evaluate_positions: Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]],
    ) -> None:
        self.evaluate_positions = evaluate_positions

    def evaluate_batch(self, requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        return self.evaluate_positions(requests)


def _legal_move_evaluator(model: torch.nn.Module, torch_device: torch.device):
    def evaluate_batch(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        if not requests:
            return []
        boards = [shogi.Board(position_sfen) for position_sfen, _legal_moves in requests]
        max_legal_move_count = max(len(legal_moves) for _position_sfen, legal_moves in requests)
        position_features = stack_shogi_position_features(
            [shogi_position_features_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
        ).to(torch_device)
        legal_move_features = torch.stack(
            [
                shogi_legal_move_features(
                    legal_moves,
                    turn=board.turn,
                    max_legal_move_count=max_legal_move_count,
                )
                for board, (_position_sfen, legal_moves) in zip(boards, requests, strict=True)
            ]
        ).to(torch_device)
        legal_move_mask = torch.zeros((len(requests), max_legal_move_count), dtype=torch.bool, device=torch_device)
        for index, (_position_sfen, legal_moves) in enumerate(requests):
            legal_move_mask[index, : len(legal_moves)] = True

        with torch.no_grad():
            if hasattr(model, "forward_policy_value"):
                logits, values = model.forward_policy_value(position_features, legal_move_features, legal_move_mask)
            else:
                logits = model(position_features, legal_move_features, legal_move_mask)
                values = model.predict_value(position_features) if hasattr(model, "predict_value") else None

        evaluations: list[PositionEvaluation] = []
        for index, (_position_sfen, legal_moves) in enumerate(requests):
            move_logits = logits[index, : len(legal_moves)]
            evaluations.append((_move_priors(move_logits, legal_moves), _value(values, index)))
        return evaluations

    return evaluate_batch


def _policy_plane_evaluator(model: torch.nn.Module, torch_device: torch.device):
    def evaluate_batch(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        if not requests:
            return []
        boards = [shogi.Board(position_sfen) for position_sfen, _legal_moves in requests]
        position_features = stack_shogi_position_features(
            [shogi_position_features_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
        ).to(torch_device)
        legal_action_mask = torch.stack([shogi_policy_plane_legal_mask(board) for board in boards]).to(torch_device)

        with torch.no_grad():
            if hasattr(model, "forward_policy_value"):
                logits, values = model.forward_policy_value(position_features, legal_action_mask)
            else:
                logits = model(position_features, legal_action_mask)
                values = model.predict_value(position_features) if hasattr(model, "predict_value") else None

        evaluations: list[PositionEvaluation] = []
        for index, (board, (_position_sfen, legal_moves)) in enumerate(zip(boards, requests, strict=True)):
            action_indices = torch.tensor(
                [shogi_policy_plane_action_index(move, turn=board.turn) for move in legal_moves],
                dtype=torch.long,
                device=torch_device,
            )
            move_logits = logits[index].index_select(0, action_indices)
            evaluations.append((_move_priors(move_logits, legal_moves), _value(values, index)))
        return evaluations

    return evaluate_batch


def _move_priors(move_logits: torch.Tensor, legal_moves: tuple[str, ...]) -> dict[str, float]:
    probabilities = torch.softmax(move_logits, dim=0).detach().cpu().tolist()
    return {move: float(probabilities[move_index]) for move_index, move in enumerate(legal_moves)}


def _value(values: torch.Tensor | None, index: int) -> float:
    if values is None:
        return 0.0
    return float(values[index].detach().cpu().item())
