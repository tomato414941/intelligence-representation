from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from intrep.problems.shogi_policy_value.generated_data_cycle import ShogiGeneratedDataTrainingCycleResult


class RunShogiGeneratedDataTrainingCycleScriptTest(unittest.TestCase):
    def test_passes_cli_arguments_to_cycle_config_and_prints_result(self) -> None:
        module = _load_script_module()
        result = ShogiGeneratedDataTrainingCycleResult(
            run_dir=Path("/tmp/cycle").resolve(),
            generated_games_jsonl=Path("/tmp/cycle/generated-games.jsonl"),
            train_games=1,
            eval_games=1,
            data_selection=Path("/tmp/cycle/data-selection.json"),
            checkpoint=Path("/tmp/cycle/checkpoint.pt"),
            best_checkpoint=Path("/tmp/cycle/best-checkpoint.pt"),
            metrics=Path("/tmp/cycle/metrics.json"),
            generation={
                "black_player": {"kind": "checkpoint", "name": "black"},
                "white_player": {"kind": "usi_engine", "name": "white-usi-engine"},
                "games": 2,
                "max_plies": 4,
                "simulations": 3,
                "evaluation_batch_size": 4,
                "mcts_move_time_limit_sec": 9.0,
            },
        )
        run_cycle = Mock(return_value=result)

        with (
            patch.object(module, "run_shogi_generated_data_training_cycle", run_cycle),
            patch.object(module, "print") as print_,
        ):
            module.main(
                [
                    "--checkpoint",
                    "source.pt",
                    "--run-dir",
                    "cycle",
                    "--arena-repo",
                    "arena",
                    "--white-player",
                    "usi-engine",
                    "--usi-command",
                    "engine-command",
                    "--usi-option",
                    "Threads=2",
                    "--usi-go-command",
                    "go nodes 2",
                    "--games",
                    "2",
                    "--concurrent-games-per-process",
                    "2",
                    "--board-backend",
                    "cshogi",
                    "--max-plies",
                    "4",
                    "--simulations",
                    "3",
                    "--evaluation-batch-size",
                    "4",
                    "--generation-worker-processes",
                    "3",
                    "--seed",
                    "11",
                    "--mcts-move-time-limit-sec",
                    "9.0",
                    "--eval-ratio",
                    "0.5",
                    "--max-steps",
                    "5",
                    "--batch-size",
                    "6",
                    "--learning-rate",
                    "0.001",
                    "--policy-loss-weight",
                    "0.7",
                    "--value-loss-weight",
                    "0.3",
                    "--allow-nonstandard-loss-weights",
                    "--device",
                    "cuda",
                    "--num-workers",
                    "2",
                ]
            )

        config = run_cycle.call_args.args[0]
        self.assertEqual(config.checkpoint, Path("source.pt"))
        self.assertEqual(config.run_dir, Path("cycle"))
        self.assertEqual(config.arena_repo, Path("arena"))
        self.assertEqual(config.black_player.kind, "checkpoint")
        self.assertEqual(config.white_player.kind, "usi_engine")
        self.assertEqual(config.white_player.usi_command, "engine-command")
        self.assertEqual(config.white_player.usi_options, ("Threads=2",))
        self.assertEqual(config.white_player.usi_go_command, "go nodes 2")
        self.assertEqual(config.games, 2)
        self.assertEqual(config.concurrent_games_per_process, 2)
        self.assertEqual(config.board_backend, "cshogi")
        self.assertEqual(config.max_plies, 4)
        self.assertEqual(config.simulations, 3)
        self.assertEqual(config.evaluation_batch_size, 4)
        self.assertEqual(config.generation_worker_processes, 3)
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.mcts_move_time_limit_sec, 9.0)
        self.assertEqual(config.eval_ratio, 0.5)
        self.assertEqual(config.max_steps, 5)
        self.assertEqual(config.batch_size, 6)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.policy_loss_weight, 0.7)
        self.assertEqual(config.value_loss_weight, 0.3)
        self.assertTrue(config.allow_nonstandard_loss_weights)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_workers, 2)
        self.assertEqual(json.loads(print_.call_args.args[0]), result.to_json())


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_generated_data_training_cycle.py"
    spec = importlib.util.spec_from_file_location("run_shogi_generated_data_training_cycle", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
