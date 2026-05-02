from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intrep.grid_world_prediction import GridStepPredictionConfig, GridStepTrainingArtifacts
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.transformer_core import SharedTransformerCore


@dataclass(frozen=True)
class GridCoreCheckpoint:
    core: SharedTransformerCore
    config: GridStepPredictionConfig
    grid_size: tuple[int, int]
    metrics: dict[str, object]


def save_grid_core_checkpoint(path: str | Path, artifacts: GridStepTrainingArtifacts) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "intrep.grid_core_checkpoint.v1",
            "core": artifacts.model.core.state_dict(),
            "config": asdict(artifacts.config),
            "grid_size": artifacts.grid_size,
            "metrics": asdict(artifacts.result),
        },
        checkpoint_path,
    )


def load_grid_core_checkpoint(
    path: str | Path,
    *,
    device: LanguageModelingTrainingDevice = "auto",
) -> GridCoreCheckpoint:
    resolved_device = resolve_training_device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    if payload.get("schema_version") != "intrep.grid_core_checkpoint.v1":
        raise ValueError("unsupported grid core checkpoint schema")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint requires config")
    config = GridStepPredictionConfig(**config_payload)
    grid_size = _grid_size_from_payload(payload.get("grid_size"))
    core = SharedTransformerCore(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(resolved_device)
    core.load_state_dict(payload["core"])
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return GridCoreCheckpoint(
        core=core,
        config=config,
        grid_size=grid_size,
        metrics=metrics,
    )


def _grid_size_from_payload(payload: object) -> tuple[int, int]:
    if isinstance(payload, (list, tuple)) and len(payload) == 2 and all(isinstance(value, int) for value in payload):
        return tuple(payload)
    raise ValueError("checkpoint requires grid_size")
