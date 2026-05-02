from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingResult


SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA = "intrep.shogi_move_choice_checkpoint.v1"


def save_shogi_move_choice_checkpoint(path: str | Path, result: ShogiMoveChoiceTrainingResult) -> None:
    torch.save(
        {
            "schema_version": SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA,
            "config": {
                "embedding_dim": result.config.embedding_dim,
                "hidden_dim": result.config.hidden_dim,
                "num_heads": result.config.num_heads,
                "num_layers": result.config.num_layers,
                "use_shared_core": result.config.use_shared_core,
                "value_loss_weight": result.config.value_loss_weight,
            },
            "model_state_dict": result.model.state_dict(),
        },
        path,
    )


def load_shogi_move_choice_checkpoint(path: str | Path, *, device: str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_MOVE_CHOICE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi move choice checkpoint schema")
    from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingConfig, build_shogi_move_choice_model

    config_payload = payload["config"]
    model = build_shogi_move_choice_model(
        ShogiMoveChoiceTrainingConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload.get("num_heads", 4)),
            num_layers=int(config_payload.get("num_layers", 1)),
            use_shared_core=bool(config_payload.get("use_shared_core", False)),
            value_loss_weight=float(config_payload.get("value_loss_weight", 0.0)),
        )
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(torch.device(device))
    model.eval()
    return model
