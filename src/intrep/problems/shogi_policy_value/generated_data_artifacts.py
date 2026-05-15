from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingCycleResult:
    run_dir: Path
    generated_games_jsonl: Path
    train_games: int
    eval_games: int
    data_selection: Path
    checkpoint: Path
    best_checkpoint: Path
    metrics: Path
    generation: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "generated_games_jsonl": str(self.generated_games_jsonl),
            "train_games": self.train_games,
            "eval_games": self.eval_games,
            "data_selection": str(self.data_selection),
            "checkpoint": str(self.checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "metrics": str(self.metrics),
            "generation": self.generation,
        }


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingLoopResult:
    run_dir: Path
    initial_checkpoint: Path
    final_checkpoint: Path
    next_checkpoint: str
    cycles: tuple[ShogiGeneratedDataTrainingCycleResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "final_checkpoint": str(self.final_checkpoint),
            "next_checkpoint": self.next_checkpoint,
            "cycles": [cycle.to_json() for cycle in self.cycles],
        }


def promoted_generated_data_checkpoint(result: ShogiGeneratedDataTrainingCycleResult, *, policy: str) -> Path:
    if policy == "best":
        return result.best_checkpoint
    if policy == "final":
        return result.checkpoint
    raise ValueError("next_checkpoint must be best or final")
