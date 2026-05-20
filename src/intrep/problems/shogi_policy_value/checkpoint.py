from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from intrep.problems.shogi_policy_value.model import (
    SHOGI_POLICY_VALUE_MODEL_ID,
    shogi_policy_value_model_spec,
    validate_shogi_policy_value_components,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingResult
from intrep.worlds.shogi.position_encoding import (
    SHOGI_POSITION_FEATURE_MANIFEST,
    SHOGI_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_POSITION_INPUT_SCHEMA_ID,
)


SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA = "intrep.problems.shogi_policy_value.checkpoint.v1"
SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX = "shogi-policy-value:sha256:"


@dataclass(frozen=True)
class ShogiPolicyValueCheckpointIdentity:
    checkpoint_id: str
    checkpoint_sha256: str
    schema_version: str
    model: str
    input: str
    core: str
    policy_output: str
    value_output: str
    input_feature_manifest_hash: str


def save_shogi_policy_value_checkpoint(path: str | Path, result: ShogiPolicyValueTrainingResult) -> None:
    save_shogi_policy_value_model_checkpoint(path, result.model, result.config)


def save_shogi_policy_value_model_checkpoint(path: str | Path, model: nn.Module, config: object) -> None:
    save_shogi_policy_value_state_checkpoint(path, model.state_dict(), config)


def save_shogi_policy_value_state_checkpoint(path: str | Path, state_dict: object, config: object) -> None:
    config_payload = _checkpoint_config_payload(config)
    checkpoint_sha256 = shogi_policy_value_checkpoint_content_sha256(
        config_payload=config_payload,
        state_dict=state_dict,
    )
    config_payload = {
        **config_payload,
        "checkpoint_id": shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256),
        "checkpoint_sha256": checkpoint_sha256,
    }
    torch.save(
        {
            "schema_version": SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA,
            "config": config_payload,
            "model_state_dict": state_dict,
        },
        path,
    )


def load_shogi_policy_value_checkpoint_identity(
    path: str | Path,
    *,
    device: str = "cpu",
) -> ShogiPolicyValueCheckpointIdentity:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    _validate_checkpoint_identity(payload)
    config = payload["config"]
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    return ShogiPolicyValueCheckpointIdentity(
        checkpoint_id=str(config["checkpoint_id"]),
        checkpoint_sha256=str(config["checkpoint_sha256"]),
        schema_version=str(payload["schema_version"]),
        model=str(config["model"]),
        input=str(config["input"]),
        core=str(config["core"]),
        policy_output=str(config["policy_output"]),
        value_output=str(config["value_output"]),
        input_feature_manifest_hash=str(config["input_feature_manifest_hash"]),
    )


def load_shogi_policy_value_checkpoint_state_dict(path: str | Path, *, device: str = "cpu") -> object:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    _validate_checkpoint_identity(payload)
    return payload["model_state_dict"]


def load_shogi_policy_value_checkpoint_training_config(path: str | Path, *, device: str = "cpu") -> object:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(payload)
    _validate_checkpoint_model_spec(payload)
    _validate_checkpoint_identity(payload)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig

    config_payload = payload["config"]
    return ShogiPolicyValueTrainingConfig(
        embedding_dim=int(config_payload["embedding_dim"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        num_heads=int(config_payload.get("num_heads", 4)),
        num_layers=int(config_payload.get("num_layers", 1)),
        input=str(config_payload["input"]),
        core=str(config_payload["core"]),
        policy_output=str(config_payload["policy_output"]),
        value_output=str(config_payload["value_output"]),
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
    _validate_checkpoint_identity(payload)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model

    config_payload = payload["config"]
    model = build_shogi_policy_value_model(
        ShogiPolicyValueTrainingConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload.get("num_heads", 4)),
            num_layers=int(config_payload.get("num_layers", 1)),
            input=str(config_payload["input"]),
            core=str(config_payload["core"]),
            policy_output=str(config_payload["policy_output"]),
            value_output=str(config_payload["value_output"]),
            value_loss_weight=float(config_payload.get("value_loss_weight", 1.0)),
            allow_nonstandard_loss_weights=bool(config_payload.get("allow_nonstandard_loss_weights", False)),
        )
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256: str) -> str:
    return f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{checkpoint_sha256}"


def shogi_policy_value_checkpoint_content_sha256(
    *,
    config_payload: dict[str, object],
    state_dict: object,
) -> str:
    hasher = hashlib.sha256()
    identity_config = {
        key: value
        for key, value in config_payload.items()
        if key not in {"checkpoint_id", "checkpoint_sha256"}
    }
    _hash_json(hasher, {"schema_version": SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA, "config": identity_config})
    _hash_state_dict(hasher, state_dict)
    return hasher.hexdigest()


def _checkpoint_config_payload(config: object) -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_POSITION_INPUT_SCHEMA_ID,
        "input_feature_manifest": SHOGI_POSITION_FEATURE_MANIFEST,
        "input_feature_manifest_hash": SHOGI_POSITION_FEATURE_MANIFEST_HASH,
        "model": SHOGI_POLICY_VALUE_MODEL_ID,
        "input": _config_str(config, "input"),
        "core": _config_str(config, "core"),
        "policy_output": _config_str(config, "policy_output"),
        "value_output": _config_str(config, "value_output"),
        "model_spec": _checkpoint_model_spec(config),
        "embedding_dim": _config_int(config, "embedding_dim"),
        "hidden_dim": _config_int(config, "hidden_dim"),
        "num_heads": _config_int(config, "num_heads"),
        "num_layers": _config_int(config, "num_layers"),
        "policy_loss_weight": _config_float(config, "policy_loss_weight"),
        "value_loss_weight": _config_float(config, "value_loss_weight"),
        "allow_nonstandard_loss_weights": _config_bool(config, "allow_nonstandard_loss_weights"),
    }


def _validate_checkpoint_input_schema_id(payload: dict[str, object]) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    if config.get("input_schema_id") != SHOGI_POSITION_INPUT_SCHEMA_ID:
        raise ValueError("unsupported shogi checkpoint input schema")
    if config.get("input_feature_manifest_hash") != SHOGI_POSITION_FEATURE_MANIFEST_HASH:
        raise ValueError("unsupported shogi checkpoint input feature manifest")
    if config.get("input_feature_manifest") != SHOGI_POSITION_FEATURE_MANIFEST:
        raise ValueError("unsupported shogi checkpoint input feature manifest")


def _validate_checkpoint_model_spec(payload: dict[str, object]) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    if config.get("model_spec") != _checkpoint_model_spec(config):
        raise ValueError("unsupported shogi checkpoint model spec")


def _validate_checkpoint_identity(payload: dict[str, object]) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    checkpoint_sha256 = config.get("checkpoint_sha256")
    checkpoint_id = config.get("checkpoint_id")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError("shogi checkpoint identity requires checkpoint_sha256")
    if checkpoint_id != shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256):
        raise ValueError("shogi checkpoint identity does not match checkpoint_sha256")
    expected_sha256 = shogi_policy_value_checkpoint_content_sha256(
        config_payload=config,
        state_dict=payload.get("model_state_dict"),
    )
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("shogi checkpoint identity does not match checkpoint contents")


def _checkpoint_model_spec(config: object) -> dict[str, object]:
    model = str(config.get("model", SHOGI_POLICY_VALUE_MODEL_ID)) if isinstance(config, dict) else SHOGI_POLICY_VALUE_MODEL_ID
    if model != SHOGI_POLICY_VALUE_MODEL_ID:
        raise ValueError(f"unsupported shogi policy/value model: {model}")
    input_name = _config_str(config, "input")
    core = _config_str(config, "core")
    policy_output = _config_str(config, "policy_output")
    value_output = _config_str(config, "value_output")
    validate_shogi_policy_value_components(
        input=input_name,
        core=core,
        policy_output=policy_output,
        value_output=value_output,
    )
    return shogi_policy_value_model_spec(
        input=input_name,
        core=core,
        policy_output=policy_output,
        value_output=value_output,
    )


def _config_str(config: object, name: str) -> str:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"shogi checkpoint config missing {name}")
        return str(config[name])
    return str(getattr(config, name))


def _config_int(config: object, name: str) -> int:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"shogi checkpoint config missing {name}")
        return int(config[name])
    return int(getattr(config, name))


def _config_float(config: object, name: str) -> float:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"shogi checkpoint config missing {name}")
        return float(config[name])
    return float(getattr(config, name))


def _config_bool(config: object, name: str) -> bool:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"shogi checkpoint config missing {name}")
        return bool(config[name])
    return bool(getattr(config, name))


def _hash_json(hasher: "hashlib._Hash", value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _hash_state_dict(hasher: "hashlib._Hash", state_dict: object) -> None:
    if not isinstance(state_dict, dict):
        raise ValueError("shogi checkpoint model_state_dict must be an object")
    keys = sorted(state_dict.keys(), key=str)
    for key in keys:
        name = str(key)
        value = state_dict[key]
        _hash_json(hasher, {"state_key": name})
        _hash_state_value(hasher, value)


def _hash_state_value(hasher: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        _hash_json(
            hasher,
            {
                "type": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
        )
        raw = tensor.numpy().tobytes()
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
        return
    _hash_json(hasher, {"type": type(value).__name__, "value": value})
