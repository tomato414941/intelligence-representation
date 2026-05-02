from __future__ import annotations

from pathlib import Path

import torch

from intrep.shogi_move_choice_model import ShogiMoveChoiceModel, ShogiMoveChoiceModelConfig
from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingResult


SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA = "intrep.shogi_move_choice_checkpoint.v1"


def save_shogi_move_choice_checkpoint(path: str | Path, result: ShogiMoveChoiceTrainingResult) -> None:
    torch.save(
        {
            "schema_version": SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA,
            "config": {
                "embedding_dim": result.model.config.embedding_dim,
                "hidden_dim": result.model.config.hidden_dim,
            },
            "model_state_dict": result.model.state_dict(),
        },
        path,
    )


def load_shogi_move_choice_checkpoint(path: str | Path, *, device: str = "cpu") -> ShogiMoveChoiceModel:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi move choice checkpoint schema")
    config_payload = payload["config"]
    model = ShogiMoveChoiceModel(
        ShogiMoveChoiceModelConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
        )
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model
