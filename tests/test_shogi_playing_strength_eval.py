from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from intrep.problems.shogi_policy_value.playing_strength_eval import (
    ShogiPlayingStrengthEvalConfig,
    build_shogi_playing_strength_eval_command,
    checkpoint_playing_strength_player,
    run_shogi_playing_strength_eval,
    usi_engine_playing_strength_player,
)


class ShogiPlayingStrengthEvalTest(unittest.TestCase):
    def test_builds_random_opening_pair_command_for_checkpoint_match(self) -> None:
        config = ShogiPlayingStrengthEvalConfig(
            arena_repo=Path("/arena"),
            output_dir=Path("/out/match"),
            player_a=checkpoint_playing_strength_player("candidate", checkpoint_id="candidate-id"),
            player_b=checkpoint_playing_strength_player("baseline", checkpoint_id="baseline-id"),
            games=64,
            start_position_seed=20260614,
            opening_plies=12,
            device="cuda",
        )

        command = build_shogi_playing_strength_eval_command(config)

        self.assertIn("/arena/scripts/evaluate_shogi_players.py", command)
        self.assertIn("--start-position-set", command)
        self.assertEqual(command[command.index("--start-position-set") + 1], "random-legal-opening")
        self.assertEqual(command[command.index("--start-position-count") + 1], "32")
        self.assertEqual(command[command.index("--start-position-seed") + 1], "20260614")
        self.assertEqual(command[command.index("--opening-plies") + 1], "12")
        self.assertEqual(command[command.index("--player-a-move-selection-profile") + 1], "max-visit")
        self.assertEqual(command[command.index("--player-b-move-selection-profile") + 1], "max-visit")
        self.assertEqual(command[command.index("--player-a-checkpoint") + 1], str(Path("candidate").resolve()))
        self.assertEqual(command[command.index("--player-b-checkpoint") + 1], str(Path("baseline").resolve()))
        self.assertNotIn("visit-sampling", command)
        self.assertEqual(command[command.index("--out") + 1], "/out/match/games.jsonl")

    def test_builds_usi_opponent_args(self) -> None:
        config = ShogiPlayingStrengthEvalConfig(
            arena_repo=Path("/arena"),
            output_dir=Path("/out/match"),
            player_a=checkpoint_playing_strength_player("candidate"),
            player_b=usi_engine_playing_strength_player(
                command="/engine/YaneuraOu",
                options=("EvalDir=/eval", "Threads=1"),
                go_command="go nodes 1000",
            ),
            games=2,
            start_position_seed=7,
        )

        command = build_shogi_playing_strength_eval_command(config)

        self.assertEqual(command[command.index("--player-b-kind") + 1], "usi_engine")
        self.assertEqual(command[command.index("--player-b-usi-command") + 1], "/engine/YaneuraOu")
        self.assertIn("--player-b-usi-option", command)
        self.assertIn("EvalDir=/eval", command)
        self.assertEqual(command[command.index("--player-b-usi-go-command") + 1], "go nodes 1000")

    def test_rejects_odd_game_count(self) -> None:
        config = ShogiPlayingStrengthEvalConfig(
            arena_repo=Path("/arena"),
            output_dir=Path("/out/match"),
            player_a=checkpoint_playing_strength_player("candidate"),
            player_b=checkpoint_playing_strength_player("baseline"),
            games=3,
            start_position_seed=7,
        )

        with self.assertRaisesRegex(ValueError, "games must be even"):
            build_shogi_playing_strength_eval_command(config)

    def test_rejects_short_max_plies(self) -> None:
        config = ShogiPlayingStrengthEvalConfig(
            arena_repo=Path("/arena"),
            output_dir=Path("/out/match"),
            player_a=checkpoint_playing_strength_player("candidate"),
            player_b=checkpoint_playing_strength_player("baseline"),
            games=2,
            start_position_seed=7,
            max_plies=128,
        )

        with self.assertRaisesRegex(ValueError, "max_plies must be at least"):
            build_shogi_playing_strength_eval_command(config)

    def test_run_writes_protocol_and_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "eval"
            config = ShogiPlayingStrengthEvalConfig(
                arena_repo=Path("/arena"),
                output_dir=output_dir,
                player_a=checkpoint_playing_strength_player("candidate"),
                player_b=checkpoint_playing_strength_player("baseline"),
                games=2,
                start_position_seed=7,
            )
            completed = Mock(stdout=json.dumps({"game_count": 2}) + "\n")

            with patch("subprocess.run", return_value=completed) as run:
                result = run_shogi_playing_strength_eval(config)

            self.assertEqual(run.call_args.kwargs["cwd"], Path("/arena"))
            self.assertEqual(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")), {"game_count": 2})
            protocol = json.loads((output_dir / "playing_strength_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["protocol"], "random_legal_opening_pair_v1")
            self.assertEqual(protocol["start_position_count"], 1)
            self.assertEqual(protocol["files"]["start_positions"], "start_positions.jsonl")
            self.assertEqual(result["summary"], {"game_count": 2})

    def test_script_requires_exactly_one_player_b_kind(self) -> None:
        module = _load_script_module()

        with self.assertRaises(SystemExit):
            module.main(
                [
                    "--output-dir",
                    "out",
                    "--player-a-checkpoint",
                    "candidate",
                    "--player-b-checkpoint",
                    "baseline",
                    "--player-b-usi-command",
                    "engine",
                    "--games",
                    "2",
                    "--start-position-seed",
                    "7",
                ]
            )

    def test_script_passes_config_to_runner(self) -> None:
        module = _load_script_module()
        run_eval = Mock(return_value={"ok": True})

        with patch.object(module, "run_shogi_playing_strength_eval", run_eval), patch.object(module, "print"):
            module.main(
                [
                    "--arena-repo",
                    "/arena",
                    "--output-dir",
                    "out",
                    "--player-a-checkpoint",
                    "candidate",
                    "--player-b-checkpoint",
                    "baseline",
                    "--games",
                    "64",
                    "--start-position-seed",
                    "20260614",
                    "--opening-plies",
                    "12",
                    "--device",
                    "cuda",
                ]
            )

        config = run_eval.call_args.args[0]
        self.assertEqual(config.arena_repo, Path("/arena"))
        self.assertEqual(config.output_dir, Path("out"))
        self.assertEqual(config.player_a.checkpoint, Path("candidate"))
        self.assertEqual(config.player_b.checkpoint, Path("baseline"))
        self.assertEqual(config.games, 64)
        self.assertEqual(config.start_position_seed, 20260614)
        self.assertEqual(config.opening_plies, 12)
        self.assertEqual(config.device, "cuda")


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_playing_strength_eval.py"
    spec = importlib.util.spec_from_file_location("run_shogi_playing_strength_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
