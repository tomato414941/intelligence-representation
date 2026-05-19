from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from intrep.problems.shogi_policy_value.model import (
    SHOGI_POLICY_VALUE_MODEL_NAMES,
    shogi_policy_value_model_spec,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingResult
from intrep.worlds.shogi.position_encoding import SHOGI_POSITION_INPUT_SCHEMA_ID


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
                "input_schema_id": SHOGI_POSITION_INPUT_SCHEMA_ID,
                "model": config.model,
                "model_spec": _checkpoint_model_spec(config),
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
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
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    return payload["model_state_dict"]


def load_shogi_policy_value_checkpoint_training_config(path: str | Path, *, device: str = "cpu") -> object:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig

    config_payload = payload["config"]
    return ShogiPolicyValueTrainingConfig(
        embedding_dim=int(config_payload["embedding_dim"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        num_heads=int(config_payload.get("num_heads", 4)),
        num_layers=int(config_payload.get("num_layers", 1)),
        model=str(config_payload["model"]),
        policy_loss_weight=float(config_payload.get("policy_loss_weight", 1.0)),
        value_loss_weight=float(config_payload.get("value_loss_weight", 1.0)),
        allow_nonstandard_loss_weights=bool(config_payload.get("allow_nonstandard_loss_weights", False)),
    )


def load_shogi_policy_value_checkpoint(path: str | Path, *, device: str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model

    config_payload = payload["config"]
    model = build_shogi_policy_value_model(
        ShogiPolicyValueTrainingConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload.get("num_heads", 4)),
            num_layers=int(config_payload.get("num_layers", 1)),
            model=str(config_payload["model"]),
            value_loss_weight=float(config_payload.get("value_loss_weight", 1.0)),
            allow_nonstandard_loss_weights=bool(config_payload.get("allow_nonstandard_loss_weights", False)),
        )
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _validate_checkpoint_input_schema_id(payload: dict[str, object]) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    if config.get("input_schema_id") != SHOGI_POSITION_INPUT_SCHEMA_ID:
        raise ValueError("unsupported shogi checkpoint input schema")


def _validate_checkpoint_model_spec(payload: dict[str, object]) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    if config.get("model_spec") != _checkpoint_model_spec(config):
        raise ValueError("unsupported shogi checkpoint model spec")


def _checkpoint_model_spec(config: object) -> dict[str, object]:
    model = _config_str(config, "model")
    if model not in SHOGI_POLICY_VALUE_MODEL_NAMES:
        raise ValueError(f"unsupported shogi policy/value model: {model}")
    return shogi_policy_value_model_spec(model)


def _config_str(config: object, name: str) -> str:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"shogi checkpoint config missing {name}")
        return str(config[name])
    return str(getattr(config, name))
