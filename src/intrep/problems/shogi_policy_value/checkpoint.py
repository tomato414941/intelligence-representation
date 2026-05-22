from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from intrep.problems.shogi_policy_value.position_input_identity import (
    shogi_position_input_identity_for_assembly_spec_id,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingResult
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_ASSEMBLY_ID,
    shogi_policy_value_assembly_spec_for_id,
)


SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA = "intrep.problems.shogi_policy_value.component_checkpoint.v1"
SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX = "shogi-policy-value:sha256:"
SHOGI_POLICY_VALUE_CHECKPOINT_MANIFEST = "manifest.json"
SHOGI_POLICY_VALUE_COMPONENT_FILES = {
    "input": "input.pt",
    "core": "core.pt",
    "policy_output": "policy_output.pt",
    "value_output": "value_output.pt",
}


@dataclass(frozen=True)
class ShogiPolicyValueCheckpointIdentity:
    checkpoint_id: str
    checkpoint_sha256: str
    schema_version: str
    assembly: str
    assembly_spec_id: str
    input_feature_manifest_hash: str


def save_shogi_policy_value_checkpoint(path: str | Path, result: ShogiPolicyValueTrainingResult) -> None:
    save_shogi_policy_value_model_checkpoint(path, result.model, result.config)


def save_shogi_policy_value_model_checkpoint(path: str | Path, model: nn.Module, config: object) -> None:
    components = _model_component_state_dicts(model)
    save_shogi_policy_value_component_checkpoint(path, components=components, config=config)


def save_shogi_policy_value_state_checkpoint(path: str | Path, state_dict: object, config: object) -> None:
    components = _split_model_state_dict(state_dict)
    save_shogi_policy_value_component_checkpoint(path, components=components, config=config)


def save_shogi_policy_value_component_checkpoint(
    path: str | Path,
    *,
    components: dict[str, object],
    config: object,
) -> None:
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    component_manifest: dict[str, dict[str, object]] = {}
    for component_name, file_name in SHOGI_POLICY_VALUE_COMPONENT_FILES.items():
        if component_name not in components:
            raise ValueError(f"shogi checkpoint missing component: {component_name}")
        component_path = checkpoint_dir / file_name
        torch.save({"component": component_name, "state_dict": components[component_name]}, component_path)
        component_manifest[component_name] = {
            "path": file_name,
            "sha256": _file_sha256(component_path),
        }

    config_payload = _checkpoint_config_payload(config)
    manifest = {
        "schema_version": SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA,
        "config": config_payload,
        "components": component_manifest,
    }
    checkpoint_sha256 = shogi_policy_value_checkpoint_content_sha256(manifest)
    manifest["config"] = {
        **config_payload,
        "checkpoint_id": shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256),
        "checkpoint_sha256": checkpoint_sha256,
    }
    _write_json(checkpoint_dir / SHOGI_POLICY_VALUE_CHECKPOINT_MANIFEST, manifest)


def load_shogi_policy_value_checkpoint_identity(
    path: str | Path,
    *,
    device: str = "cpu",
) -> ShogiPolicyValueCheckpointIdentity:
    del device
    manifest = _load_checkpoint_manifest(path)
    _validate_checkpoint_manifest(Path(path), manifest)
    config = _checkpoint_config(manifest)
    return ShogiPolicyValueCheckpointIdentity(
        checkpoint_id=str(config["checkpoint_id"]),
        checkpoint_sha256=str(config["checkpoint_sha256"]),
        schema_version=str(manifest["schema_version"]),
        assembly=str(config["assembly"]),
        assembly_spec_id=str(config["assembly_spec_id"]),
        input_feature_manifest_hash=str(config["input_feature_manifest_hash"]),
    )


def load_shogi_policy_value_checkpoint_state_dict(path: str | Path, *, device: str = "cpu") -> object:
    manifest = _load_checkpoint_manifest(path)
    _validate_checkpoint_manifest(Path(path), manifest)
    return _merge_component_state_dicts(_load_component_state_dicts(Path(path), manifest, device=device))


def load_shogi_policy_value_checkpoint_training_config(path: str | Path, *, device: str = "cpu") -> object:
    del device
    manifest = _load_checkpoint_manifest(path)
    _validate_checkpoint_manifest(Path(path), manifest)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig

    config_payload = _checkpoint_config(manifest)
    return ShogiPolicyValueTrainingConfig(
        embedding_dim=int(config_payload["embedding_dim"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        num_heads=int(config_payload["num_heads"]),
        num_layers=int(config_payload["num_layers"]),
        assembly_spec_id=str(config_payload["assembly_spec_id"]),
        policy_loss_weight=float(config_payload["policy_loss_weight"]),
        value_loss_weight=float(config_payload["value_loss_weight"]),
        allow_nonstandard_loss_weights=bool(config_payload["allow_nonstandard_loss_weights"]),
    )


def load_shogi_policy_value_checkpoint(path: str | Path, *, device: str = "cpu") -> nn.Module:
    manifest = _load_checkpoint_manifest(path)
    _validate_checkpoint_manifest(Path(path), manifest)
    from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model

    config_payload = _checkpoint_config(manifest)
    model = build_shogi_policy_value_model(
        ShogiPolicyValueTrainingConfig(
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload["num_heads"]),
            num_layers=int(config_payload["num_layers"]),
            assembly_spec_id=str(config_payload["assembly_spec_id"]),
            policy_loss_weight=float(config_payload["policy_loss_weight"]),
            value_loss_weight=float(config_payload["value_loss_weight"]),
            allow_nonstandard_loss_weights=bool(config_payload["allow_nonstandard_loss_weights"]),
        )
    )
    model.load_state_dict(load_shogi_policy_value_checkpoint_state_dict(path, device=device), strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256: str) -> str:
    return f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{checkpoint_sha256}"


def shogi_policy_value_checkpoint_content_sha256(manifest: dict[str, object]) -> str:
    identity_manifest = {
        **manifest,
        "config": {
            key: value
            for key, value in _checkpoint_config(manifest).items()
            if key not in {"checkpoint_id", "checkpoint_sha256"}
        },
    }
    return hashlib.sha256(
        json.dumps(identity_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _model_component_state_dicts(model: nn.Module) -> dict[str, object]:
    return {
        "input": {
            "input": model.encoder.input.state_dict(),
            "attention_logit_bias": model.encoder.attention_logit_bias.state_dict(),
        },
        "core": model.encoder.core.state_dict(),
        "policy_output": model.policy_output.state_dict(),
        "value_output": model.value_output.state_dict(),
    }


def _split_model_state_dict(state_dict: object) -> dict[str, object]:
    if not isinstance(state_dict, dict):
        raise ValueError("shogi checkpoint state_dict must be an object")
    prefixes = {
        "encoder.input.": ("input", "input"),
        "encoder.attention_logit_bias.": ("input", "attention_logit_bias"),
        "encoder.core.": ("core", None),
        "policy_output.": ("policy_output", None),
        "value_output.": ("value_output", None),
    }
    components: dict[str, Any] = {
        "input": {"input": {}, "attention_logit_bias": {}},
        "core": {},
        "policy_output": {},
        "value_output": {},
    }
    for key, value in state_dict.items():
        name = str(key)
        for prefix, (component, subcomponent) in prefixes.items():
            if name.startswith(prefix):
                stripped = name.removeprefix(prefix)
                if subcomponent is None:
                    components[component][stripped] = value
                else:
                    components[component][subcomponent][stripped] = value
                break
        else:
            raise ValueError(f"unsupported shogi checkpoint state_dict key: {name}")
    return components


def _merge_component_state_dicts(components: dict[str, object]) -> dict[str, object]:
    input_component = components["input"]
    if not isinstance(input_component, dict):
        raise ValueError("shogi checkpoint input component must be an object")
    return {
        **_prefix_state_dict("encoder.input.", input_component["input"]),
        **_prefix_state_dict("encoder.attention_logit_bias.", input_component["attention_logit_bias"]),
        **_prefix_state_dict("encoder.core.", components["core"]),
        **_prefix_state_dict("policy_output.", components["policy_output"]),
        **_prefix_state_dict("value_output.", components["value_output"]),
    }


def _prefix_state_dict(prefix: str, state_dict: object) -> dict[str, object]:
    if not isinstance(state_dict, dict):
        raise ValueError("shogi checkpoint component state_dict must be an object")
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def _checkpoint_config_payload(config: object) -> dict[str, object]:
    assembly_spec = _checkpoint_assembly_spec(config)
    return {
        **shogi_position_input_identity_for_assembly_spec_id(str(assembly_spec["assembly_spec_id"])),
        "assembly": SHOGI_POLICY_VALUE_ASSEMBLY_ID,
        "assembly_spec_id": assembly_spec["assembly_spec_id"],
        "assembly_spec": assembly_spec,
        "embedding_dim": _config_int(config, "embedding_dim"),
        "hidden_dim": _config_int(config, "hidden_dim"),
        "num_heads": _config_int(config, "num_heads"),
        "num_layers": _config_int(config, "num_layers"),
        "policy_loss_weight": _config_float(config, "policy_loss_weight"),
        "value_loss_weight": _config_float(config, "value_loss_weight"),
        "allow_nonstandard_loss_weights": _config_bool(config, "allow_nonstandard_loss_weights"),
    }


def _load_checkpoint_manifest(path: str | Path) -> dict[str, object]:
    checkpoint_dir = Path(path)
    manifest_path = checkpoint_dir / SHOGI_POLICY_VALUE_CHECKPOINT_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shogi checkpoint manifest must be an object")
    return payload


def _validate_checkpoint_manifest(checkpoint_dir: Path, manifest: dict[str, object]) -> None:
    if manifest.get("schema_version") != SHOGI_POLICY_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported shogi policy value checkpoint schema")
    _validate_checkpoint_input_schema_id(manifest)
    _validate_checkpoint_assembly_spec(manifest)
    _validate_checkpoint_components(checkpoint_dir, manifest)
    _validate_checkpoint_identity(manifest)


def _validate_checkpoint_input_schema_id(manifest: dict[str, object]) -> None:
    config = _checkpoint_config(manifest)
    expected = shogi_position_input_identity_for_assembly_spec_id(_config_str(config, "assembly_spec_id"))
    if config.get("input_schema_id") != expected["input_schema_id"]:
        raise ValueError("unsupported shogi checkpoint input schema")
    if config.get("input_feature_manifest_hash") != expected["input_feature_manifest_hash"]:
        raise ValueError("unsupported shogi checkpoint input feature manifest")
    if config.get("input_feature_manifest") != expected["input_feature_manifest"]:
        raise ValueError("unsupported shogi checkpoint input feature manifest")


def _validate_checkpoint_assembly_spec(manifest: dict[str, object]) -> None:
    config = _checkpoint_config(manifest)
    assembly_spec = _checkpoint_assembly_spec(config)
    if config.get("assembly_spec") != assembly_spec:
        raise ValueError("unsupported shogi checkpoint assembly spec")
    if config.get("assembly_spec_id") != assembly_spec["assembly_spec_id"]:
        raise ValueError("unsupported shogi checkpoint assembly spec")


def _validate_checkpoint_components(checkpoint_dir: Path, manifest: dict[str, object]) -> None:
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise ValueError("shogi checkpoint components must be an object")
    for component_name, file_name in SHOGI_POLICY_VALUE_COMPONENT_FILES.items():
        component = components.get(component_name)
        if not isinstance(component, dict):
            raise ValueError(f"shogi checkpoint missing component: {component_name}")
        if component.get("path") != file_name:
            raise ValueError(f"unsupported shogi checkpoint component path: {component_name}")
        component_path = checkpoint_dir / file_name
        if not component_path.is_file():
            raise ValueError(f"shogi checkpoint component file not found: {component_name}")
        if component.get("sha256") != _file_sha256(component_path):
            raise ValueError(f"shogi checkpoint component identity mismatch: {component_name}")


def _validate_checkpoint_identity(manifest: dict[str, object]) -> None:
    config = _checkpoint_config(manifest)
    checkpoint_sha256 = config.get("checkpoint_sha256")
    checkpoint_id = config.get("checkpoint_id")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError("shogi checkpoint identity requires checkpoint_sha256")
    if checkpoint_id != shogi_policy_value_checkpoint_id_from_sha256(checkpoint_sha256):
        raise ValueError("shogi checkpoint identity does not match checkpoint_sha256")
    if checkpoint_sha256 != shogi_policy_value_checkpoint_content_sha256(manifest):
        raise ValueError("shogi checkpoint identity does not match checkpoint contents")


def _load_component_state_dicts(
    checkpoint_dir: Path,
    manifest: dict[str, object],
    *,
    device: str,
) -> dict[str, object]:
    components = manifest["components"]
    if not isinstance(components, dict):
        raise ValueError("shogi checkpoint components must be an object")
    loaded: dict[str, object] = {}
    for component_name in SHOGI_POLICY_VALUE_COMPONENT_FILES:
        component = components[component_name]
        if not isinstance(component, dict):
            raise ValueError(f"shogi checkpoint missing component: {component_name}")
        payload = torch.load(checkpoint_dir / str(component["path"]), map_location=torch.device(device), weights_only=False)
        if not isinstance(payload, dict) or payload.get("component") != component_name:
            raise ValueError(f"unsupported shogi checkpoint component payload: {component_name}")
        loaded[component_name] = payload["state_dict"]
    return loaded


def _checkpoint_config(manifest: dict[str, object]) -> dict[str, object]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("shogi checkpoint config must be an object")
    return config


def _checkpoint_assembly_spec(config: object) -> dict[str, object]:
    assembly = _config_str(config, "assembly") if isinstance(config, dict) else SHOGI_POLICY_VALUE_ASSEMBLY_ID
    if assembly != SHOGI_POLICY_VALUE_ASSEMBLY_ID:
        raise ValueError(f"unsupported shogi policy/value assembly: {assembly}")
    return shogi_policy_value_assembly_spec_for_id(_config_str(config, "assembly_spec_id"))


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


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
