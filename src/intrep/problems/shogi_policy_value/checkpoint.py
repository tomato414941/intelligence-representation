from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingResult


SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA = "intrep.problems.shogi_policy_value.checkpoint.v1"


def save_shogi_policy_value_checkpoint(path: str | Path, result: ShogiPolicyValueTrainingResult) -> None:
    save_shogi_policy_value_model_checkpoint(path, result.model, result.config)


def save_shogi_policy_value_model_checkpoint(path: str | Path, model: nn.Module, config: object) -> None:
    save_shogi_policy_value_state_checkpoint(path, model.state_dict(), config)


def save_shogi_policy_value_state_checkpoint(path: str | Path, state_dict: object, config: object) -> None:
    torch.save(
        {
            "schema_version": SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA,
            "config": {
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
                "use_shared_core": config.use_shared_core,
                "policy_loss_weight": config.policy_loss_weight,
                "value_loss_weight": config.value_loss_weight,
                "allow_nonstandard_loss_weights": config.allow_nonstandard_loss_weights,
            },
            "model_state_dict": state_dict,
        },
        path,
    )


def load_shogi_policy_value_checkpoint_state_dict(path: str | Path, *, device: str = "cpu") -> object:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    return payload["model_state_dict"]


def load_shogi_policy_value_checkpoint_training_config(path: str | Path, *, device: str = "cpu") -> object:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig

    config_payload = payload["config"]
    return ShogiPolicyValueTrainingConfig(
        embedding_dim=int(config_payload["embedding_dim"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        num_heads=int(config_payload.get("num_heads", 4)),
        num_layers=int(config_payload.get("num_layers", 1)),
        use_shared_core=bool(config_payload.get("use_shared_core", False)),
        policy_loss_weight=float(config_payload.get("policy_loss_weight", 1.0)),
        value_loss_weight=float(config_payload.get("value_loss_weight", 0.0)),
        allow_nonstandard_loss_weights=bool(config_payload.get("allow_nonstandard_loss_weights", False)),
    )


def load_shogi_policy_value_checkpoint(path: str | Path, *, device: str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    from intrep.problems.shogi_policy_value.model import adapt_shogi_policy_value_state_dict_for_model
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model

    config_payload = payload["config"]
    model = build_shogi_policy_value_model(
        ShogiPolicyValueTrainingConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload.get("num_heads", 4)),
            num_layers=int(config_payload.get("num_layers", 1)),
            use_shared_core=bool(config_payload.get("use_shared_core", False)),
            value_loss_weight=float(config_payload.get("value_loss_weight", 0.0)),
            allow_nonstandard_loss_weights=bool(config_payload.get("allow_nonstandard_loss_weights", False)),
        )
    )
    model.load_state_dict(adapt_shogi_policy_value_state_dict_for_model(payload["model_state_dict"], model), strict=True)
    model.to(torch.device(device))
    model.eval()
    return model
