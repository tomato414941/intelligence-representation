from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shogi

from intrep.problems.shogi_policy_value.generated_data_cycle import (
    ShogiGeneratedDataTrainingCycleConfig,
    run_shogi_generated_data_training_cycle,
)
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )


class ShogiGeneratedDataCycleTest(unittest.TestCase):
    def test_rejects_invalid_config_before_running_commands(self) -> None:
        with patch("intrep.problems.shogi_policy_value.generated_data_cycle.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "games"):
                run_shogi_generated_data_training_cycle(
                    ShogiGeneratedDataTrainingCycleConfig(
                        checkpoint=Path("source.pt"),
                        run_dir=Path("cycle"),
                        games=0,
                    )
                )

        run.assert_not_called()

    def test_requires_yaneuraou_command_for_yaneuraou_opponent(self) -> None:
        with self.assertRaisesRegex(ValueError, "yaneuraou"):
            run_shogi_generated_data_training_cycle(
                ShogiGeneratedDataTrainingCycleConfig(
                    checkpoint=Path("source.pt"),
                    run_dir=Path("cycle"),
                    opponent="yaneuraou",
                )
            )

    def test_runs_one_cycle_through_generation_split_and_training_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "cycle"
            arena_repo = root / "arena"
            arena_repo.mkdir()

            def fake_run(command: list[str], **_kwargs: object) -> None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(
                        out_path,
                        [
                            _record(("7g7f", "3c3d"), "black"),
                            _record(("2g2f", "8c8d"), "white"),
                        ],
                    )

            with patch("intrep.problems.shogi_policy_value.generated_data_cycle.subprocess.run", side_effect=fake_run) as run:
                result = run_shogi_generated_data_training_cycle(
                    ShogiGeneratedDataTrainingCycleConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        arena_repo=arena_repo,
                        games=2,
                        max_plies=4,
                        simulations=3,
                        evaluation_batch_size=4,
                        mcts_move_time_limit_sec=9.0,
                        max_steps=5,
                        batch_size=2,
                    )
                )

            dataset = json.loads((run_dir / "data-selection.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["target_construction"]["policy"], "chosen_move")
            self.assertEqual(dataset["target_construction"]["value"], "winner")
            self.assertEqual(dataset["train_sources"][0]["kind"], "game_records_jsonl")
            self.assertEqual(dataset["eval_sources"][0]["kind"], "game_records_jsonl")
            self.assertTrue((run_dir / "train-games.jsonl").exists())
            self.assertTrue((run_dir / "eval-games.jsonl").exists())
            self.assertEqual(run.call_count, 2)
            generate_command = run.call_args_list[0].args[0]
            self.assertEqual(generate_command[generate_command.index("--black-kind") + 1], "checkpoint")
            self.assertEqual(generate_command[generate_command.index("--white-kind") + 1], "checkpoint")
            self.assertEqual(generate_command[generate_command.index("--black-checkpoint-simulations") + 1], "3")
            self.assertEqual(generate_command[generate_command.index("--white-checkpoint-simulations") + 1], "3")
            self.assertEqual(generate_command[generate_command.index("--black-checkpoint-evaluation-batch-size") + 1], "4")
            self.assertEqual(generate_command[generate_command.index("--white-checkpoint-evaluation-batch-size") + 1], "4")
            self.assertEqual(generate_command[generate_command.index("--black-checkpoint-move-time-limit-sec") + 1], "9.0")
            self.assertEqual(generate_command[generate_command.index("--white-checkpoint-move-time-limit-sec") + 1], "9.0")
            train_command = run.call_args_list[1].args[0]
            self.assertIn("intrep.train_shogi_policy_value", train_command)
            self.assertEqual(train_command[train_command.index("--init-checkpoint-path") + 1], str(checkpoint_path))
            self.assertIn("--value-loss-weight", train_command)
            self.assertIn("1.0", train_command)
            self.assertEqual(
                result.generation,
                {
                    "opponent": "self",
                    "games": 2,
                    "max_plies": 4,
                    "simulations": 3,
                    "evaluation_batch_size": 4,
                    "mcts_move_time_limit_sec": 9.0,
                },
            )

    def test_passes_yaneuraou_generation_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "cycle"
            arena_repo = root / "arena"
            arena_repo.mkdir()

            def fake_run(command: list[str], **_kwargs: object) -> None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(
                        out_path,
                        [
                            _record(("7g7f", "3c3d"), "black"),
                            _record(("2g2f", "8c8d"), "white"),
                        ],
                    )

            with patch("intrep.problems.shogi_policy_value.generated_data_cycle.subprocess.run", side_effect=fake_run) as run:
                run_shogi_generated_data_training_cycle(
                    ShogiGeneratedDataTrainingCycleConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        arena_repo=arena_repo,
                        opponent="yaneuraou",
                        yaneuraou="engine-command",
                        engine_go_command="go nodes 2",
                        games=2,
                    )
                )

            generate_command = run.call_args_list[0].args[0]
            self.assertEqual(generate_command[generate_command.index("--black-kind") + 1], "checkpoint")
            self.assertEqual(generate_command[generate_command.index("--white-kind") + 1], "yaneuraou")
            self.assertEqual(generate_command[generate_command.index("--white-yaneuraou-command") + 1], "engine-command")
            self.assertEqual(generate_command[generate_command.index("--white-yaneuraou-go-command") + 1], "go nodes 2")
            self.assertNotIn("--black-checkpoint-move-time-limit-sec", generate_command)
            self.assertNotIn("--white-checkpoint-move-time-limit-sec", generate_command)


if __name__ == "__main__":
    unittest.main()
