from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from intrep.problems.shogi_policy_value.generated_data_cycle import (
    ShogiGeneratedDataTrainingCycleConfig,
    ShogiGeneratedDataTrainingLoopConfig,
    run_shogi_generated_data_training_cycle,
    run_shogi_generated_data_training_loop,
)
from intrep.problems.shogi_policy_value.online_replay import (
    ShogiGeneratedExperienceSource,
    ShogiOnlineReplayConfig,
    ShogiOnlineReplayTrainingBudget,
    run_shogi_online_replay,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    checkpoint_generated_player,
    usi_engine_generated_player,
)
from intrep.problems.shogi_policy_value.examples import TensorizedShogiPolicyValueSample
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingMetrics,
    ShogiPolicyValueTrainingProgress,
    ShogiPolicyValueTrainingResult,
    build_shogi_policy_value_model,
)
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        winner=winner,
    )


def _self_play_source(*, games: int, name: str = "self-play") -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=checkpoint_generated_player("black"),
        white_player=checkpoint_generated_player("white"),
    )


def _checkpoint_vs_usi_source(*, games: int, name: str = "checkpoint-vs-usi") -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=checkpoint_generated_player("checkpoint"),
        white_player=usi_engine_generated_player(
            name="usi-engine",
            command="engine",
            options=("Threads=2",),
            go_command="go nodes 4",
            read_timeout_seconds=31,
        ),
    )


def _usi_vs_checkpoint_source(*, games: int, name: str = "usi-vs-checkpoint") -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=usi_engine_generated_player(
            name="usi-engine",
            command="engine",
            options=("Threads=2",),
            go_command="go nodes 4",
            read_timeout_seconds=31,
        ),
        white_player=checkpoint_generated_player("checkpoint"),
    )


def _write_training_eval_bundle(bundle_dir: Path) -> Path:
    write_shogi_game_records_jsonl(bundle_dir / "train-games.jsonl", [_record(("9g9f", "9c9d"), "black")])
    write_shogi_game_records_jsonl(bundle_dir / "eval-games.jsonl", [_record(("1g1f", "1c1d"), "white")])
    (bundle_dir / "data-selection.json").write_text(
        json.dumps(
            {
                "name": bundle_dir.name,
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
    return bundle_dir / "data-selection.json"


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
            with self.assertRaisesRegex(ValueError, "generator_gate_games"):
                run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=Path("source.pt"),
                        run_dir=Path("online"),
                        training_eval_data_selection=Path("eval/data-selection.json"),
                        generator_gate_games=0,
                    )
                )

            run.assert_not_called()

            with self.assertRaisesRegex(ValueError, "generator_gate_worker_processes"):
                run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=Path("source.pt"),
                        run_dir=Path("online"),
                        training_eval_data_selection=Path("eval/data-selection.json"),
                        generator_gate_worker_processes=0,
                    )
                )

            run.assert_not_called()

            with self.assertRaisesRegex(ValueError, "games"):
                run_shogi_generated_data_training_cycle(
                    ShogiGeneratedDataTrainingCycleConfig(
                        checkpoint=Path("source.pt"),
                        run_dir=Path("cycle"),
                        games=0,
                    )
                )

        run.assert_not_called()

    def test_requires_usi_command_for_usi_engine_player(self) -> None:
        with self.assertRaisesRegex(ValueError, "usi"):
            run_shogi_generated_data_training_cycle(
                ShogiGeneratedDataTrainingCycleConfig(
                    checkpoint=Path("source.pt"),
                    run_dir=Path("cycle"),
                    white_player=usi_engine_generated_player(name="engine", command=""),
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
            self.assertEqual(generate_command[generate_command.index("--black-move-selection-temperature") + 1], "1.0")
            self.assertEqual(generate_command[generate_command.index("--white-move-selection-temperature") + 1], "1.0")
            self.assertEqual(generate_command[generate_command.index("--black-move-selection-temperature-plies") + 1], "40")
            self.assertEqual(generate_command[generate_command.index("--white-move-selection-temperature-plies") + 1], "40")
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
                    "black_player": {
                        "kind": "checkpoint",
                        "name": "black",
                        "move_selection_profile": "self-play",
                        "move_selection_temperature": 1.0,
                        "move_selection_temperature_plies": 40,
                    },
                    "white_player": {
                        "kind": "checkpoint",
                        "name": "white",
                        "move_selection_profile": "self-play",
                        "move_selection_temperature": 1.0,
                        "move_selection_temperature_plies": 40,
                    },
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

    def test_passes_usi_generation_options(self) -> None:
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
                return None

            with patch("intrep.problems.shogi_policy_value.generated_data_cycle.subprocess.run", side_effect=fake_run) as run:
                run_shogi_generated_data_training_cycle(
                    ShogiGeneratedDataTrainingCycleConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        arena_repo=arena_repo,
                        white_player=usi_engine_generated_player(
                            name="engine",
                            command="engine-command",
                            options=("Threads=2",),
                            go_command="go nodes 2",
                        ),
                        games=2,
                        device="cuda",
                    )
                )

            generate_command = run.call_args_list[0].args[0]
            self.assertEqual(generate_command[generate_command.index("--black-kind") + 1], "checkpoint")
            self.assertEqual(generate_command[generate_command.index("--white-kind") + 1], "usi_engine")
            self.assertEqual(generate_command[generate_command.index("--white-usi-command") + 1], "engine-command")
            self.assertEqual(generate_command[generate_command.index("--white-usi-option") + 1], "Threads=2")
            self.assertEqual(generate_command[generate_command.index("--white-usi-go-command") + 1], "go nodes 2")
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
                return None

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
                return None

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
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")
            train_batches: list[int] = []
            gate_commands: list[list[str]] = []

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
                    return None
                if any(item.endswith("evaluate_shogi_players.py") for item in command):
                    gate_commands.append(command)
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": 2,
                                "player_a_wins": 2,
                                "player_a_losses": 0,
                                "draws": 0,
                                "average_plies": 2,
                                "illegal_move_count": 0,
                            }
                        )
                        + "\n",
                    )
                return None

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                self.assertIsInstance(examples[0], TensorizedShogiPolicyValueSample)
                self.assertIsInstance(eval_examples[0], TensorizedShogiPolicyValueSample)
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_run),
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
                        iterations=2,
                        replay_capacity=4,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(
                            sampled_examples_per_iteration=3,
                            batch_size=6,
                            target_sample_passes=2.0,
                            max_optimizer_steps=5,
                        ),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=2),),
                    )
                )

            self.assertEqual(len(result.iterations), 2)
            self.assertEqual(train_batches, [3, 3])
            self.assertEqual(result.iterations[0].appended_examples, 4)
            self.assertEqual(result.iterations[0].replay_size, 4)
            self.assertEqual(result.iterations[0].sampled_examples, 3)
            self.assertIsNone(result.iterations[0].experience_store_append)
            self.assertEqual(result.stop_reason, None)
            self.assertEqual(len(gate_commands), 1)
            self.assertEqual(gate_commands[0][gate_commands[0].index("--match-worker-processes") + 1], "4")
            gate_summary = json.loads((run_dir / "iteration-0002" / "generator-gate-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_summary["player_a_wins"], 2)
            self.assertEqual(gate_summary["player_a_losses"], 0)
            metrics = json.loads(result.iterations[0].metrics.read_text(encoding="utf-8"))
            self.assertEqual(metrics["sampled_examples"], 3)
            self.assertEqual(metrics["sampled_examples_per_iteration"], 3)
            self.assertEqual(metrics["training_batch_size"], 6)
            self.assertEqual(metrics["target_sample_passes"], 2.0)
            self.assertEqual(metrics["optimizer_steps_per_iteration"], 1)
            self.assertEqual(metrics["max_optimizer_steps_per_iteration"], 5)
            self.assertEqual(metrics["effective_sample_passes"], 2.0)

    def test_online_replay_stops_when_generator_candidate_loses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_record(("7g7f", "3c3d"), "black")])
                    return None
                if any(item.endswith("evaluate_shogi_players.py") for item in command):
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": 2,
                                "player_a_wins": 0,
                                "player_a_losses": 2,
                                "draws": 0,
                                "average_plies": 2,
                                "illegal_move_count": 0,
                            }
                        )
                        + "\n",
                    )
                return None

            with (
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", return_value=_training_result(ShogiPolicyValueTrainingConfig())),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_checkpoint"),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_state_checkpoint"),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=3,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=2),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                    )
                )

            self.assertEqual(len(result.iterations), 1)
            self.assertEqual(result.stop_reason, "generator_candidate_lost")
            self.assertEqual(result.stopped_iteration_index, 2)
            self.assertEqual(run.call_count, 2)
            self.assertFalse((run_dir / "iteration-0002" / "generated-games.jsonl").exists())

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
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=8),
                        experience_store_dir=store_dir,
                        replay_seed_data_selection=bundle_dir / "data-selection.json",
                        training_eval_data_selection=bundle_dir / "data-selection.json",
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=2),),
                    )
                )

            self.assertEqual(result.preloaded_examples, 2)
            self.assertEqual(result.training_eval_examples, 2)
            self.assertEqual(result.experience_store_dir, store_dir)
            self.assertEqual(result.replay_seed_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(result.training_eval_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(train_batches, [6])
            self.assertEqual(eval_batches, [2])
            self.assertEqual(result.iterations[0].replay_size, 6)
            self.assertIsNotNone(result.iterations[0].experience_store_append)
            self.assertEqual(result.iterations[0].experience_store_append["added_games"], 2)
            self.assertEqual(load_shogi_game_records_jsonl(store_dir / "games.jsonl"), generated_records)
            metrics = json.loads((run_dir / "iteration-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["preloaded_examples"], 2)
            self.assertEqual(metrics["training_eval_examples"], 2)
            self.assertEqual(metrics["generated_holdout_examples"], 0)
            self.assertEqual(metrics["training_eval_source"], "fixed_data_selection")
            self.assertEqual(metrics["experience_store_dir"], str(store_dir))
            self.assertEqual(metrics["replay_seed_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["training_eval_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["experience_store_append"]["total_games"], 2)
            self.assertEqual(metrics["generation_summary"]["game_count"], 2)
            self.assertTrue((run_dir / "iteration-0001" / "generation-summary.json").exists())

    def test_online_replay_generates_multiple_experience_sources_per_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    if command[command.index("--white-kind") + 1] == "usi_engine":
                        records = [_record(("2g2f", "8c8d"), "white")]
                    elif command[command.index("--black-kind") + 1] == "usi_engine":
                        records = [_record(("7g7f", "3c3d"), "black")]
                    else:
                        records = [_record(("7g7f", "3c3d"), "black")]
                    write_shogi_game_records_jsonl(out_path, records)
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": len(records),
                                "average_plies": 2,
                                "end_reasons": {"game_over": len(records)},
                                "black_wins": 1 if records[0].winner == "black" else 0,
                                "white_wins": 1 if records[0].winner == "white" else 0,
                                "draws": 0,
                                "generation_wall_time_sec": 1.0,
                                "plies_per_sec": 2.0,
                            }
                        )
                        + "\n",
                    )
                return None

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", return_value=_training_result(ShogiPolicyValueTrainingConfig())),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_checkpoint"),
                patch("intrep.problems.shogi_policy_value.online_replay.save_shogi_policy_value_state_checkpoint"),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=8),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(
                            _self_play_source(games=1),
                            _checkpoint_vs_usi_source(games=1),
                            _usi_vs_checkpoint_source(games=1),
                        ),
                        concurrent_games_per_process=8,
                    )
                )

            self.assertEqual(result.iterations[0].appended_examples, 6)
            self.assertEqual(run.call_count, 3)
            self_play_command = run.call_args_list[0].args[0]
            self.assertEqual(self_play_command[self_play_command.index("--seed") + 1], "7")
            self.assertEqual(
                self_play_command[self_play_command.index("--concurrent-games-per-process") + 1],
                "8",
            )
            usi_command = run.call_args_list[1].args[0]
            self.assertEqual(usi_command[usi_command.index("--seed") + 1], "8")
            self.assertEqual(
                usi_command[usi_command.index("--concurrent-games-per-process") + 1],
                "1",
            )
            self.assertEqual(usi_command[usi_command.index("--white-kind") + 1], "usi_engine")
            self.assertEqual(usi_command[usi_command.index("--white-usi-command") + 1], "engine")
            self.assertEqual(usi_command[usi_command.index("--white-usi-option") + 1], "Threads=2")
            self.assertEqual(usi_command[usi_command.index("--white-usi-read-timeout-seconds") + 1], "31")
            reversed_usi_command = run.call_args_list[2].args[0]
            self.assertEqual(reversed_usi_command[reversed_usi_command.index("--black-kind") + 1], "usi_engine")
            self.assertEqual(reversed_usi_command[reversed_usi_command.index("--black-usi-command") + 1], "engine")
            self.assertEqual(reversed_usi_command[reversed_usi_command.index("--white-kind") + 1], "checkpoint")
            records = load_shogi_game_records_jsonl(run_dir / "iteration-0001" / "generated-games.jsonl")
            self.assertEqual(len(records), 3)
            summary = json.loads((run_dir / "iteration-0001" / "generation-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["game_count"], 3)
            self.assertEqual([source["name"] for source in summary["sources"]], ["self-play", "checkpoint-vs-usi", "usi-vs-checkpoint"])
            self.assertEqual(summary["sources"][1]["black_player"]["kind"], "checkpoint")
            self.assertEqual(summary["sources"][1]["white_player"]["kind"], "usi_engine")
            self.assertEqual(summary["sources"][2]["black_player"]["kind"], "usi_engine")
            self.assertEqual(summary["sources"][2]["white_player"]["kind"], "checkpoint")

    def test_online_replay_passes_training_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")
            captured_config: ShogiPolicyValueTrainingConfig | None = None

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
                nonlocal captured_config
                captured_config = config
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
                run_shogi_online_replay(
                    ShogiOnlineReplayConfig(
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(
                            sampled_examples_per_iteration=8,
                            batch_size=4,
                            target_sample_passes=3.0,
                        ),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                        training_config=ShogiPolicyValueTrainingConfig(
                            learning_rate=0.01,
                            weight_decay=0.02,
                            policy_loss_weight=0.7,
                            value_loss_weight=0.3,
                            allow_nonstandard_loss_weights=True,
                            device="cpu",
                            max_train_eval_examples=5,
                            max_eval_examples=6,
                            log_every=7,
                            num_workers=0,
                            pin_memory=True,
                            progress_every=8,
                            eval_every=3,
                            early_stopping_patience=2,
                            seed=13,
                        ),
                        seed=13,
                    )
                )

            self.assertIsNotNone(captured_config)
            assert captured_config is not None
            self.assertEqual(captured_config.max_steps, 3)
            self.assertEqual(captured_config.batch_size, 4)
            self.assertEqual(captured_config.learning_rate, 0.01)
            self.assertEqual(captured_config.weight_decay, 0.02)
            self.assertEqual(captured_config.seed, 13)
            self.assertEqual(captured_config.policy_loss_weight, 0.7)
            self.assertEqual(captured_config.value_loss_weight, 0.3)
            self.assertTrue(captured_config.allow_nonstandard_loss_weights)
            self.assertEqual(captured_config.max_train_eval_examples, 5)
            self.assertEqual(captured_config.max_eval_examples, 6)
            self.assertEqual(captured_config.log_every, 7)
            self.assertTrue(captured_config.pin_memory)
            self.assertEqual(captured_config.progress_every, 8)
            self.assertEqual(captured_config.eval_every, 3)
            self.assertEqual(captured_config.early_stopping_patience, 2)

    def test_online_replay_reports_training_progress_with_iteration_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_record(("7g7f", "3c3d"), "black")])

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                result = _training_result(config)
                self.assertIsNotNone(progress_callback)
                assert progress_callback is not None
                progress_callback(
                    ShogiPolicyValueTrainingProgress(
                        step=2,
                        max_steps=4,
                        loss=1.25,
                        elapsed_seconds=3.5,
                        data_wait_seconds=0.1,
                        forward_backward_seconds=0.2,
                        optimizer_seconds=0.03,
                        model=result.model,
                        config=config,
                    )
                )
                return result

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
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_shogi_online_replay(
                        ShogiOnlineReplayConfig(
                            checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(
                            sampled_examples_per_iteration=2,
                            batch_size=1,
                            target_sample_passes=2.0,
                        ),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                        training_config=ShogiPolicyValueTrainingConfig(progress_every=2),
                        )
                    )

            output = stdout.getvalue()
            self.assertIn("online_replay_training_progress", output)
            self.assertIn("iteration=1", output)
            self.assertIn("step=2/4", output)
            self.assertIn("loss=1.250000", output)
            self.assertIn("replay_size=2", output)
            self.assertIn("sampled_examples=2", output)
            self.assertIn("training_eval_examples=2", output)

    def test_online_replay_skips_training_until_min_replay_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")
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
                        iterations=2,
                        replay_capacity=8,
                        min_replay_size=5,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=3),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=2),),
                    )
                )

            self.assertEqual(train_batches, [3])
            self.assertTrue(result.iterations[0].training_skipped)
            self.assertEqual(result.iterations[0].sampled_examples, 0)
            self.assertEqual(result.iterations[0].best_checkpoint, checkpoint_path)
            self.assertFalse(result.iterations[1].training_skipped)
            first_metrics = json.loads((run_dir / "iteration-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(first_metrics["training_skipped"])
            self.assertEqual(first_metrics["skip_reason"], "min_replay_size")
            self.assertIsNone(first_metrics["effective_sample_passes"])


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
