from __future__ import annotations

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class RunShogiPlayingEvalScriptTest(unittest.TestCase):
    def test_builds_checkpoint_vs_yaneuraou_command(self) -> None:
        module = _load_script_module()
        args = Namespace(
            checkpoint=Path("model.pt"),
            out=Path("runs/eval/games.jsonl"),
            opponent_kind="yaneuraou",
            opponent_checkpoint=None,
            yaneuraou="engine",
            engine_go_command="go nodes 2",
            games=4,
            max_plies=320,
            simulations=128,
            evaluation_batch_size=64,
            move_time_limit_sec=10.0,
            device="cuda",
            board_backend="cshogi",
        )

        command = module.build_shogi_playing_eval_command(args)

        self.assertIn("scripts/evaluate_shogi_players.py", command)
        self.assertEqual(command[command.index("--player-kind") + 1], "checkpoint")
        self.assertEqual(command[command.index("--player-checkpoint-profile") + 1], "evaluation")
        self.assertEqual(command[command.index("--player-checkpoint-simulations") + 1], "128")
        self.assertEqual(command[command.index("--player-checkpoint-evaluation-batch-size") + 1], "64")
        self.assertEqual(command[command.index("--player-checkpoint-device") + 1], "cuda")
        self.assertEqual(command[command.index("--player-checkpoint-board-backend") + 1], "cshogi")
        self.assertEqual(command[command.index("--player-checkpoint-move-time-limit-sec") + 1], "10.0")
        self.assertEqual(command[command.index("--opponent-kind") + 1], "yaneuraou")
        self.assertEqual(command[command.index("--opponent-yaneuraou-command") + 1], "engine")
        self.assertEqual(command[command.index("--opponent-yaneuraou-go-command") + 1], "go nodes 2")
        self.assertEqual(command[command.index("--games") + 1], "4")
        self.assertEqual(command[command.index("--max-plies") + 1], "320")
        self.assertEqual(command[command.index("--out") + 1], str(Path("runs/eval/games.jsonl").resolve()))

    def test_builds_checkpoint_vs_checkpoint_command(self) -> None:
        module = _load_script_module()
        args = Namespace(
            checkpoint=Path("candidate.pt"),
            out=Path("games.jsonl"),
            opponent_kind="checkpoint",
            opponent_checkpoint=Path("baseline.pt"),
            yaneuraou=None,
            engine_go_command="go nodes 1",
            games=2,
            max_plies=320,
            simulations=16,
            evaluation_batch_size=8,
            move_time_limit_sec=None,
            device="cpu",
            board_backend="cshogi",
        )

        command = module.build_shogi_playing_eval_command(args)

        self.assertEqual(command[command.index("--opponent-kind") + 1], "checkpoint")
        self.assertEqual(command[command.index("--opponent-checkpoint-profile") + 1], "evaluation")
        self.assertEqual(command[command.index("--opponent-checkpoint-policy") + 1], "mcts")
        self.assertEqual(command[command.index("--opponent-checkpoint-simulations") + 1], "16")
        self.assertNotIn("--opponent-yaneuraou-command", command)

    def test_main_runs_arena_evaluator(self) -> None:
        module = _load_script_module()

        with patch.object(module.subprocess, "run") as run:
            module.main(
                [
                    "--arena-repo",
                    ".",
                    "--checkpoint",
                    "candidate.pt",
                    "--out",
                    "games.jsonl",
                    "--opponent-kind",
                    "deterministic_legal",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(run.call_args.kwargs["cwd"], Path(".").resolve())
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--opponent-kind") + 1], "deterministic_legal")


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_playing_eval.py"
    spec = importlib.util.spec_from_file_location("run_shogi_playing_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
