from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from intrep.experience.multimodal.records import write_selection
from intrep.problems.multimodal_agent.runtime import checkpoint_identity, rollout
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.worlds.gridworld.multimodal import MultimodalNavigationWorld


def collect_and_learn(
    checkpoint: Path, selections: Sequence[Path], output: Path, *, rounds: int = 1,
    episodes_per_round: int = 16, learning_steps: int = 100, horizon: int = 8,
    seed: int = 95001, epsilon: float = 0.15, device: str = "auto",
) -> Path:
    """Alternate real interaction and explicit mixed replay, at episode boundaries."""
    if min(rounds, episodes_per_round, learning_steps) < 1:
        raise ValueError("interaction and learning budgets must be positive")
    if output.exists():
        raise FileExistsError("cycle output already exists; use a new directory")
    output.mkdir(parents=True)
    retained = list(selections)
    evaluation_worlds = {
        row["world_id"] for selection in retained for row in json.loads(selection.read_text())["episodes"]
        if row["split"] != "train"
    }
    candidate_seed = seed
    history = []
    current = checkpoint
    for round_index in range(rounds):
        model, payload = load_checkpoint(current, device=device)
        identity = checkpoint_identity(current)
        round_root = output / f"round-{round_index:03d}"
        source_paths, summaries = [], []
        while len(source_paths) < episodes_per_round:
            world = MultimodalNavigationWorld(candidate_seed, horizon=horizon)
            episode_seed = candidate_seed
            candidate_seed += 1
            if world.world_id in evaluation_worlds:
                continue
            _, summary = rollout(model, checkpoint_id=identity, seed=episode_seed,
                                 source_root=round_root / "episodes",
                                 display_root=round_root / f"replay-{len(source_paths):03d}",
                                 horizon=horizon, epsilon=epsilon)
            source_paths.append(Path(summary["source_path"]))
            summaries.append(summary)
        selection = write_selection(round_root, {"train": source_paths})
        retained.append(selection)
        config = MultimodalTrainingConfig(**{**payload["config"], "model": model.config})
        config = dataclasses.replace(config, steps=learning_steps, seed=config.seed + round_index + 1)
        del model
        current = train(retained, round_root / "learning", config, device=device, initialize=current)
        record = {"round": round_index, "actor_checkpoint_sha256": identity,
                  "learned_checkpoint_sha256": checkpoint_identity(current),
                  "new_episodes": [summary["episode_id"] for summary in summaries],
                  "mean_reward": sum(summary["total_reward"] for summary in summaries) / len(summaries),
                  "replay_selection": str(selection), "checkpoint": str(current)}
        history.append(record)
        (round_root / "rollouts.json").write_text(json.dumps({"episodes": summaries}, ensure_ascii=False, indent=2) + "\n")
        (output / "cycles.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    return current
