from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shogi

from intrep.problems.shogi_policy_value.generated_data_cycle import (
    ShogiGeneratedDataTrainingCycleConfig,
    ShogiGeneratedDataTrainingLoopConfig,
    ShogiOnlineReplayConfig,
    run_shogi_online_replay,
    run_shogi_generated_data_training_loop,
    run_shogi_generated_data_training_cycle,
)
from intrep.problems.shogi_policy_value.examples import TensorizedShogiPolicyValueSample
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingMetrics,
    ShogiPolicyValueTrainingResult,
    build_shogi_policy_value_model,
)
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
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
    def test_rejects_invalid_loop_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "cycles"):
            run_shogi_generated_data_training_loop(
                ShogiGeneratedDataTrainingLoopConfig(
                    checkpoint=Path("source.pt"),
                    run_dir=Path("loop"),
                    cycles=0,
                )
            )

        with self.assertRaisesRegex(ValueError, "next_checkpoint"):
            run_shogi_generated_data_training_loop(
                ShogiGeneratedDataTrainingLoopConfig(
                    checkpoint=Path("source.pt"),
                    run_dir=Path("loop"),
                    next_checkpoint="latest",
                )
            )

    def test_rejects_invalid_config_before_running_commands(self) -> None:
        with patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run") as run:
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

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
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
                        concurrent_games_per_process=2,
                        max_plies=4,
                        simulations=3,
                        evaluation_batch_size=4,
                        generation_worker_processes=3,
                        seed=11,
                        mcts_move_time_limit_sec=9.0,
                        max_steps=5,
                        batch_size=2,
                        device="cuda",
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
            self.assertEqual(generate_command[generate_command.index("--black-checkpoint-id") + 1], "source")
            self.assertEqual(generate_command[generate_command.index("--white-checkpoint-id") + 1], "source")
            self.assertEqual(generate_command[generate_command.index("--concurrent-games-per-process") + 1], "2")
            self.assertEqual(generate_command[generate_command.index("--generation-worker-processes") + 1], "3")
            self.assertEqual(generate_command[generate_command.index("--seed") + 1], "11")
            self.assertEqual(generate_command[generate_command.index("--progress-every-plies") + 1], "0")
            self.assertEqual(generate_command[generate_command.index("--black-mcts-simulations") + 1], "3")
            self.assertEqual(generate_command[generate_command.index("--white-mcts-simulations") + 1], "3")
            self.assertEqual(generate_command[generate_command.index("--black-move-selection-profile") + 1], "self-play")
            self.assertEqual(generate_command[generate_command.index("--white-move-selection-profile") + 1], "self-play")
            self.assertEqual(generate_command[generate_command.index("--black-mcts-evaluation-batch-size") + 1], "4")
            self.assertEqual(generate_command[generate_command.index("--white-mcts-evaluation-batch-size") + 1], "4")
            self.assertEqual(generate_command[generate_command.index("--black-device") + 1], "cuda")
            self.assertEqual(generate_command[generate_command.index("--white-device") + 1], "cuda")
            self.assertEqual(generate_command[generate_command.index("--black-board-backend") + 1], "cshogi")
            self.assertEqual(generate_command[generate_command.index("--white-board-backend") + 1], "cshogi")
            self.assertEqual(generate_command[generate_command.index("--board-backend") + 1], "cshogi")
            self.assertEqual(generate_command[generate_command.index("--black-mcts-move-time-limit-sec") + 1], "9.0")
            self.assertEqual(generate_command[generate_command.index("--white-mcts-move-time-limit-sec") + 1], "9.0")
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
                    "concurrent_games_per_process": 2,
                    "generation_progress_every_plies": 0,
                    "board_backend": "cshogi",
                    "max_plies": 4,
                    "simulations": 3,
                    "evaluation_batch_size": 4,
                    "generation_worker_processes": 3,
                    "seed": 11,
                    "checkpoint_device": "cuda",
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
                        device="cuda",
                    )
                )

            generate_command = run.call_args_list[0].args[0]
            self.assertEqual(generate_command[generate_command.index("--black-kind") + 1], "checkpoint")
            self.assertEqual(generate_command[generate_command.index("--white-kind") + 1], "yaneuraou")
            self.assertEqual(generate_command[generate_command.index("--white-yaneuraou-command") + 1], "engine-command")
            self.assertEqual(generate_command[generate_command.index("--white-yaneuraou-go-command") + 1], "go nodes 2")
            self.assertEqual(generate_command[generate_command.index("--black-device") + 1], "cuda")
            self.assertEqual(generate_command[generate_command.index("--black-board-backend") + 1], "cshogi")
            self.assertNotIn("--white-device", generate_command)
            self.assertNotIn("--white-board-backend", generate_command)
            self.assertNotIn("--black-mcts-move-time-limit-sec", generate_command)
            self.assertNotIn("--white-mcts-move-time-limit-sec", generate_command)

    def test_runs_multi_cycle_loop_using_best_checkpoint_as_next_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "loop"
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
                result = run_shogi_generated_data_training_loop(
                    ShogiGeneratedDataTrainingLoopConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        cycles=2,
                        next_checkpoint="best",
                        arena_repo=arena_repo,
                        games=2,
                        max_steps=1,
                    )
                )

            self.assertEqual(len(result.cycles), 2)
            self.assertEqual(result.next_checkpoint, "best")
            self.assertEqual(result.cycles[0].run_dir, run_dir.resolve() / "cycle-0001")
            self.assertEqual(result.cycles[1].run_dir, run_dir.resolve() / "cycle-0002")
            self.assertEqual(result.final_checkpoint, result.cycles[1].best_checkpoint)
            first_train_command = run.call_args_list[1].args[0]
            second_generate_command = run.call_args_list[2].args[0]
            self.assertEqual(first_train_command[first_train_command.index("--init-checkpoint-path") + 1], str(checkpoint_path))
            self.assertEqual(
                second_generate_command[second_generate_command.index("--black-checkpoint") + 1],
                str(result.cycles[0].best_checkpoint.resolve()),
            )

    def test_multi_cycle_loop_can_promote_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "loop"
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
                result = run_shogi_generated_data_training_loop(
                    ShogiGeneratedDataTrainingLoopConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        cycles=2,
                        next_checkpoint="final",
                        arena_repo=arena_repo,
                        games=2,
                        max_steps=1,
                    )
                )

            second_generate_command = run.call_args_list[2].args[0]
            self.assertEqual(result.next_checkpoint, "final")
            self.assertEqual(result.final_checkpoint, result.cycles[1].checkpoint)
            self.assertEqual(
                second_generate_command[second_generate_command.index("--black-checkpoint") + 1],
                str(result.cycles[0].checkpoint.resolve()),
            )

    def test_online_replay_appends_generated_examples_and_trains_from_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            train_batches: list[int] = []

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

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                self.assertIsInstance(examples[0], TensorizedShogiPolicyValueSample)
                self.assertIsInstance(eval_examples[0], TensorizedShogiPolicyValueSample)
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_checkpoint"),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_state_checkpoint"),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        cycles=2,
                        replay_capacity=4,
                        replay_sample_size=3,
                        min_replay_size=1,
                        arena_repo=arena_repo,
                        games=2,
                        max_steps=1,
                    )
                )

            self.assertEqual(len(result.cycles), 2)
            self.assertEqual(train_batches, [2, 3])
            self.assertEqual(result.cycles[0].appended_examples, 2)
            self.assertEqual(result.cycles[0].replay_size, 2)
            self.assertEqual(result.cycles[0].sampled_examples, 2)
            self.assertIsNone(result.cycles[0].experience_store_append)
            self.assertEqual(result.cycles[1].appended_examples, 2)
            self.assertEqual(result.cycles[1].replay_size, 4)
            self.assertEqual(result.cycles[1].sampled_examples, 3)
            second_generate_command = run.call_args_list[1].args[0]
            self.assertEqual(
                second_generate_command[second_generate_command.index("--black-checkpoint") + 1],
                str(result.cycles[0].best_checkpoint.resolve()),
            )

    def test_online_replay_seeds_replay_from_bundle_train_split_and_appends_store_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            store_dir = root / "store"
            bundle_dir = root / "bundle"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            seed_train_record = _record(("9g9f", "9c9d"), "black")
            seed_eval_record = _record(("1g1f", "1c1d"), "white")
            generated_records = [
                _record(("7g7f", "3c3d"), "black"),
                _record(("2g2f", "8c8d"), "white"),
            ]
            write_shogi_game_records_jsonl(bundle_dir / "train-games.jsonl", [seed_train_record])
            write_shogi_game_records_jsonl(bundle_dir / "eval-games.jsonl", [seed_eval_record])
            (bundle_dir / "data-selection.json").write_text(
                json.dumps(
                    {
                        "name": "seed-bundle",
                        "objective": "shogi move-choice policy/value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "analysis_sources": [],
                        "train_sources": [{"kind": "game_records_jsonl", "path": "train-games.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            train_batches: list[int] = []
            eval_batches: list[int] = []

            def fake_run(command: list[str], **_kwargs: object) -> None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, generated_records)
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps({"game_count": 2, "generation_wall_time_sec": 1.5}) + "\n",
                    )
                return None

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                eval_batches.append(len(eval_examples))
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_checkpoint"),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_state_checkpoint"),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        cycles=1,
                        replay_capacity=8,
                        replay_sample_size=8,
                        min_replay_size=1,
                        experience_store_dir=store_dir,
                        replay_seed_data_selection=bundle_dir / "data-selection.json",
                        training_eval_data_selection=bundle_dir / "data-selection.json",
                        arena_repo=arena_repo,
                        games=2,
                        max_steps=1,
                    )
                )

            self.assertEqual(result.preloaded_examples, 2)
            self.assertEqual(result.fixed_eval_examples, 2)
            self.assertEqual(result.experience_store_dir, store_dir)
            self.assertEqual(result.replay_seed_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(result.training_eval_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(train_batches, [4])
            self.assertEqual(eval_batches, [2])
            self.assertEqual(result.cycles[0].replay_size, 4)
            self.assertIsNotNone(result.cycles[0].experience_store_append)
            self.assertEqual(result.cycles[0].experience_store_append["added_games"], 2)
            self.assertEqual(load_shogi_game_records_jsonl(store_dir / "games.jsonl"), generated_records)
            metrics = json.loads((run_dir / "cycle-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["preloaded_examples"], 2)
            self.assertEqual(metrics["fixed_eval_examples"], 2)
            self.assertEqual(metrics["generated_eval_examples"], 2)
            self.assertEqual(metrics["training_eval_source"], "fixed")
            self.assertEqual(metrics["experience_store_dir"], str(store_dir))
            self.assertEqual(metrics["replay_seed_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["training_eval_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["experience_store_append"]["total_games"], 2)
            self.assertEqual(metrics["generation_summary"]["game_count"], 2)
            self.assertTrue((run_dir / "cycle-0001" / "generation-summary.json").exists())

    def test_online_replay_skips_training_until_min_replay_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            train_batches: list[int] = []

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

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_checkpoint"),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_state_checkpoint"),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        cycles=2,
                        replay_capacity=4,
                        replay_sample_size=3,
                        min_replay_size=3,
                        arena_repo=arena_repo,
                        games=2,
                        max_steps=1,
                    )
                )

            self.assertEqual(train_batches, [3])
            self.assertTrue(result.cycles[0].training_skipped)
            self.assertEqual(result.cycles[0].sampled_examples, 0)
            self.assertEqual(result.cycles[0].best_checkpoint, checkpoint_path)
            self.assertFalse(result.cycles[1].training_skipped)
            first_metrics = json.loads((run_dir / "cycle-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(first_metrics["training_skipped"])
            self.assertEqual(first_metrics["skip_reason"], "min_replay_size")
            second_generate_command = run.call_args_list[1].args[0]
            self.assertEqual(
                second_generate_command[second_generate_command.index("--black-checkpoint") + 1],
                str(checkpoint_path.resolve()),
            )


def _training_result(config: ShogiPolicyValueTrainingConfig) -> ShogiPolicyValueTrainingResult:
    return ShogiPolicyValueTrainingResult(
        model=build_shogi_policy_value_model(
            ShogiPolicyValueTrainingConfig(
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            )
        ),
        config=config,
        metrics=ShogiPolicyValueTrainingMetrics(
            train_case_count=1,
            eval_case_count=1,
            initial_loss=1.0,
            initial_value_loss=None,
            final_loss=0.5,
            accuracy=0.0,
            top_3_accuracy=0.0,
            top_5_accuracy=0.0,
            mean_reciprocal_rank=0.0,
            mean_correct_move_rank=0.0,
            value_loss=None,
            eval_loss=None,
            initial_eval_loss=None,
            eval_accuracy=None,
            initial_eval_accuracy=None,
            eval_top_3_accuracy=None,
            eval_top_5_accuracy=None,
            eval_mean_reciprocal_rank=None,
            eval_mean_correct_move_rank=None,
            eval_value_loss=None,
            initial_eval_value_loss=None,
            best_eval_loss=None,
            best_eval_step=None,
            max_steps=config.max_steps,
            actual_steps=1,
            stopped_early=False,
            stopped_step=None,
            early_stopping_patience=None,
        ),
        best_model_state_dict=None,
    )


if __name__ == "__main__":
    unittest.main()
