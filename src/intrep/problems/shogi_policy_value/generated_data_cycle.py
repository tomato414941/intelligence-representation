from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingCycleConfig:
    checkpoint: Path
    run_dir: Path
    arena_repo: Path = Path("../shogi-arena-agent")
    opponent: str = "self"
    yaneuraou: str | None = None
    engine_go_command: str = "go nodes 1"
    games: int = 4
    max_plies: int = 80
    simulations: int = 16
    evaluation_batch_size: int = 1
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0


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


def run_shogi_generated_data_training_cycle(
    config: ShogiGeneratedDataTrainingCycleConfig,
) -> ShogiGeneratedDataTrainingCycleResult:
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    games_jsonl = run_dir / "generated-games.jsonl"
    train_jsonl = run_dir / "train-games.jsonl"
    eval_jsonl = run_dir / "eval-games.jsonl"
    data_selection_json = run_dir / "data-selection.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best-checkpoint.pt"
    metrics_path = run_dir / "metrics.json"

    run_generate_games(
        arena_repo=config.arena_repo,
        checkpoint=config.checkpoint,
        opponent=config.opponent,
        yaneuraou=config.yaneuraou,
        engine_go_command=config.engine_go_command,
        out=games_jsonl,
        games=config.games,
        max_plies=config.max_plies,
        simulations=config.simulations,
        evaluation_batch_size=config.evaluation_batch_size,
        mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
    )
    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=games_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        eval_ratio=config.eval_ratio,
    )
    write_data_selection(data_selection_json, train_jsonl=train_jsonl, eval_jsonl=eval_jsonl)
    run_training(
        data_selection_json=data_selection_json,
        init_checkpoint_path=config.checkpoint,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        device=config.device,
        num_workers=config.num_workers,
    )
    return ShogiGeneratedDataTrainingCycleResult(
        run_dir=run_dir,
        generated_games_jsonl=games_jsonl,
        train_games=train_count,
        eval_games=eval_count,
        data_selection=data_selection_json,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        metrics=metrics_path,
        generation={
            "opponent": config.opponent,
            "games": config.games,
            "max_plies": config.max_plies,
            "simulations": config.simulations,
            "evaluation_batch_size": config.evaluation_batch_size,
            "mcts_move_time_limit_sec": config.mcts_move_time_limit_sec,
        },
    )


def run_generate_games(
    *,
    arena_repo: Path,
    checkpoint: Path,
    opponent: str,
    yaneuraou: str | None,
    engine_go_command: str,
    out: Path,
    games: int,
    max_plies: int,
    simulations: int,
    evaluation_batch_size: int,
    mcts_move_time_limit_sec: float | None,
) -> None:
    if opponent == "yaneuraou" and not yaneuraou:
        raise SystemExit("--yaneuraou is required when --opponent yaneuraou")

    command = [
        "uv",
        "run",
        "python",
        "scripts/generate_shogi_games.py",
        "--black-kind",
        "checkpoint",
        "--black-checkpoint",
        str(checkpoint.resolve()),
        "--black-checkpoint-policy",
        "mcts",
        "--black-checkpoint-simulations",
        str(simulations),
        "--black-checkpoint-evaluation-batch-size",
        str(evaluation_batch_size),
        "--games",
        str(games),
        "--max-plies",
        str(max_plies),
        "--out",
        str(out),
    ]
    if mcts_move_time_limit_sec is not None:
        command.extend(["--black-checkpoint-move-time-limit-sec", str(mcts_move_time_limit_sec)])
    if opponent == "yaneuraou":
        command.extend(
            [
                "--white-kind",
                "yaneuraou",
                "--white-yaneuraou-command",
                yaneuraou or "",
                "--white-yaneuraou-go-command",
                engine_go_command,
            ]
        )
    else:
        command.extend(
            [
                "--white-kind",
                "checkpoint",
                "--white-checkpoint",
                str(checkpoint.resolve()),
                "--white-checkpoint-policy",
                "mcts",
                "--white-checkpoint-simulations",
                str(simulations),
                "--white-checkpoint-evaluation-batch-size",
                str(evaluation_batch_size),
            ]
        )
        if mcts_move_time_limit_sec is not None:
            command.extend(["--white-checkpoint-move-time-limit-sec", str(mcts_move_time_limit_sec)])
    subprocess.run(command, cwd=arena_repo.resolve(), check=True)


def write_data_selection(path: Path, *, train_jsonl: Path, eval_jsonl: Path) -> None:
    payload = {
        "name": path.parent.name,
        "objective": "shogi policy/value from generated game records",
        "target_construction": {
            "policy": "chosen_move",
            "policy_temperature_cp": 100.0,
            "policy_mate_cp": 100000.0,
            "value": "winner",
            "score_cp_scale": 600.0,
        },
        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_jsonl)}],
        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_jsonl)}],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_training(
    *,
    data_selection_json: Path,
    init_checkpoint_path: Path,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    metrics_path: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    device: str,
    num_workers: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "intrep.train_shogi_policy_value",
        "--data-selection",
        str(data_selection_json),
        "--init-checkpoint-path",
        str(init_checkpoint_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--best-checkpoint-path",
        str(best_checkpoint_path),
        "--metrics-path",
        str(metrics_path),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--policy-loss-weight",
        str(policy_loss_weight),
        "--value-loss-weight",
        str(value_loss_weight),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]
    subprocess.run(command, check=True)
