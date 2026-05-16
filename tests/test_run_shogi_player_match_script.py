from __future__ import annotations

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class RunShogiPlayerMatchScriptTest(unittest.TestCase):
    def test_builds_checkpoint_vs_usi_command(self) -> None:
        module = _load_script_module()
        args = Namespace(
            player_a_kind="checkpoint",
            player_a_checkpoint=Path("model.pt"),
            player_a_usi_command=None,
            player_a_usi_option=[],
            player_a_usi_go_command="go nodes 1",
            player_b_kind="usi",
            player_b_checkpoint=None,
            player_b_usi_command="engine",
            player_b_usi_option=["Threads=2"],
            player_b_usi_go_command="go nodes 2",
            out=Path("runs/eval/games.jsonl"),
            games=4,
            max_plies=320,
            simulations=128,
            evaluation_batch_size=64,
            move_time_limit_sec=10.0,
            move_selection_profile="self-play",
            device="cuda",
            board_backend="cshogi",
        )

        command = module.build_shogi_player_match_command(args)

        self.assertIn("scripts/evaluate_shogi_players.py", command)
        self.assertEqual(command[command.index("--player-a-kind") + 1], "checkpoint")
        self.assertEqual(command[command.index("--player-a-move-selection-profile") + 1], "self-play")
        self.assertEqual(command[command.index("--player-a-mcts-simulations") + 1], "128")
        self.assertEqual(command[command.index("--player-a-mcts-evaluation-batch-size") + 1], "64")
        self.assertEqual(command[command.index("--player-a-device") + 1], "cuda")
        self.assertEqual(command[command.index("--player-a-board-backend") + 1], "cshogi")
        self.assertEqual(command[command.index("--player-a-mcts-move-time-limit-sec") + 1], "10.0")
        self.assertEqual(command[command.index("--player-b-kind") + 1], "usi")
        self.assertEqual(command[command.index("--player-b-usi-command") + 1], "engine")
        self.assertEqual(command[command.index("--player-b-usi-option") + 1], "Threads=2")
        self.assertEqual(command[command.index("--player-b-usi-go-command") + 1], "go nodes 2")
        self.assertEqual(command[command.index("--games") + 1], "4")
        self.assertEqual(command[command.index("--max-plies") + 1], "320")
        self.assertEqual(command[command.index("--out") + 1], str(Path("runs/eval/games.jsonl").resolve()))

    def test_builds_checkpoint_vs_checkpoint_command(self) -> None:
        module = _load_script_module()
        args = Namespace(
            player_a_kind="checkpoint",
            player_a_checkpoint=Path("a.pt"),
            player_a_usi_command=None,
            player_a_usi_option=[],
            player_a_usi_go_command="go nodes 1",
            player_b_kind="checkpoint",
            player_b_checkpoint=Path("b.pt"),
            player_b_usi_command=None,
            player_b_usi_option=[],
            player_b_usi_go_command="go nodes 1",
            out=Path("games.jsonl"),
            games=2,
            max_plies=320,
            simulations=16,
            evaluation_batch_size=8,
            move_time_limit_sec=None,
            move_selection_profile="evaluation",
            device="cpu",
            board_backend="cshogi",
        )

        command = module.build_shogi_player_match_command(args)

        self.assertEqual(command[command.index("--player-a-checkpoint") + 1], str(Path("a.pt").resolve()))
        self.assertEqual(command[command.index("--player-b-kind") + 1], "checkpoint")
        self.assertEqual(command[command.index("--player-b-checkpoint") + 1], str(Path("b.pt").resolve()))
        self.assertEqual(command[command.index("--player-b-move-selection-profile") + 1], "evaluation")
        self.assertEqual(command[command.index("--player-b-move-selector") + 1], "mcts")
        self.assertEqual(command[command.index("--player-b-mcts-simulations") + 1], "16")
        self.assertNotIn("--player-b-usi-command", command)

    def test_main_runs_arena_evaluator(self) -> None:
        module = _load_script_module()

        with patch.object(module.subprocess, "run") as run:
            module.main(
                [
                    "--arena-repo",
                    ".",
                    "--player-a-kind",
                    "checkpoint",
                    "--player-a-checkpoint",
                    "a.pt",
                    "--player-b-kind",
                    "deterministic_legal",
                    "--out",
                    "games.jsonl",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(run.call_args.kwargs["cwd"], Path(".").resolve())
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--player-a-kind") + 1], "checkpoint")
        self.assertEqual(
            run.call_args.args[0][run.call_args.args[0].index("--player-b-kind") + 1],
            "deterministic_legal",
        )


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_player_match.py"
    spec = importlib.util.spec_from_file_location("run_shogi_player_match", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
