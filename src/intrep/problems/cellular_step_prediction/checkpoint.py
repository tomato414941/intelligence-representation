from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intrep.core.training_utils import TrainingDevice, resolve_training_device
from intrep.problems.cellular_step_prediction.training import (
    CellularStepPredictionConfig,
    CellularStepTrainingArtifacts,
)
from intrep.representation.assemblies.cellular_step_prediction import CellularStepPredictionModel
from intrep.worlds.cellular.world import CellularRule

_SCHEMA = "intrep.cellular_step_prediction_checkpoint.v1"


@dataclass(frozen=True)
class CellularStepPredictionCheckpoint:
    model: CellularStepPredictionModel
    config: CellularStepPredictionConfig
    grid_size: tuple[int, int]
    rule: CellularRule


def save_cellular_step_checkpoint(
    path: str | Path,
    artifacts: CellularStepTrainingArtifacts,
    *,
    rule: CellularRule,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": _SCHEMA,
            "model": artifacts.model.state_dict(),
            "config": asdict(artifacts.config),
            "grid_size": artifacts.grid_size,
            "rule": {"birth": sorted(rule.birth), "survival": sorted(rule.survival)},
        },
        checkpoint_path,
    )


def load_cellular_step_checkpoint(
    path: str | Path,
    *,
    device: TrainingDevice = "auto",
) -> CellularStepPredictionCheckpoint:
    resolved_device = resolve_training_device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    if payload.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported cellular step prediction checkpoint schema")
    config = CellularStepPredictionConfig(**payload["config"])
    height, width = payload["grid_size"]
    model = CellularStepPredictionModel(
        height=height,
        width=width,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(resolved_device)
    model.load_state_dict(payload["model"])
    rule_payload = payload["rule"]
    return CellularStepPredictionCheckpoint(
        model=model,
        config=config,
        grid_size=(height, width),
        rule=CellularRule(
            birth=frozenset(rule_payload["birth"]),
            survival=frozenset(rule_payload["survival"]),
        ),
    )
