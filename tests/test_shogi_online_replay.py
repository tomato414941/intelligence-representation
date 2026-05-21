from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from intrep.problems.shogi_policy_value.online_replay import (
    ShogiGeneratedExperienceSource,
    ShogiOnlineReplayConfig,
    ShogiOnlineReplayTrainingBudget,
    _sample_replay_seed_examples_from_selection,
    _sample_replay_seed_samples,
    _load_replay_seed_selection,
    run_shogi_online_replay,
)
from intrep.problems.shogi_policy_value.tensor_cache import build_shogi_policy_value_tensor_cache
from intrep.problems.shogi_policy_value.generated_game_production import (
    checkpoint_generated_player,
    _run_generation_command,
    usi_engine_generated_player,
)
from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint_identity,
    save_shogi_policy_value_state_checkpoint,
)
from intrep.problems.shogi_policy_value.samples import LegalMovePolicyValueTensorSample
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingMetrics,
    ShogiPolicyValueTrainingProgress,
    ShogiPolicyValueTrainingResult,
    build_shogi_policy_value_model,
)
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
)
from intrep.domains.shogi.game_record import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiMoveRecord,
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


def _mcts_record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    record = _record(moves, winner)
    return replace(
        record,
        moves=tuple(
            ShogiMoveRecord(
                action_usi=move.action_usi,
                decision_telemetry=ShogiDecisionTelemetry(
                    search_evidence={"mcts_root_child_visit_counts": {move.action_usi: 3}},
                ),
            )
            for move in record.moves
        ),
    )


def _mcts_record_with_actors(
    moves: tuple[str, ...],
    winner: str | None,
    *,
    black_actor: ShogiActorSpec,
    white_actor: ShogiActorSpec,
) -> ShogiGameRecord:
    return replace(
        _mcts_record(moves, winner),
        black_actor=black_actor,
        white_actor=white_actor,
    )


def _self_play_source(*, games: int, name: str = "self-play") -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=checkpoint_generated_player("black"),
        white_player=checkpoint_generated_player("white"),
        policy_target_construction="mcts_visit_counts",
        value_target_construction="winner",
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
        policy_target_construction="chosen_move",
        value_target_construction="winner",
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
        policy_target_construction="chosen_move",
        value_target_construction="winner",
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


class ShogiOnlineReplayTest(unittest.TestCase):
    def test_replay_seed_examples_are_sampled_from_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = _write_training_eval_bundle(root / "bundle")

            sampled = _sample_replay_seed_examples_from_selection(
                _load_replay_seed_selection(selection_path),
                sample_count=1,
                seed=7,
            )

        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0].game_index, 0)
        self.assertIn(sampled[0].ply_index, {0, 1})

    def test_replay_seed_samples_use_tensor_cache_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = _write_training_eval_bundle(root / "bundle")
            build_shogi_policy_value_tensor_cache(data_selection_path=selection_path, shard_games=1)

            sampled = _sample_replay_seed_samples(
                _load_replay_seed_selection(selection_path),
                sample_count=1,
                seed=7,
            )

        self.assertEqual(len(sampled), 1)
        self.assertIsInstance(sampled[0], LegalMovePolicyValueTensorSample)

    def test_rejects_invalid_config_before_running_commands(self) -> None:
        with patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command") as run:
            with self.assertRaisesRegex(ValueError, "generator_gate_games"):
                run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=Path("source.pt"),
                        run_dir=Path("online"),
                        training_eval_data_selection=Path("eval/data-selection.json"),
                        generator_gate_games=0,
                    )
                )

            run.assert_not_called()

            with self.assertRaisesRegex(ValueError, "generator_gate_worker_processes"):
                run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=Path("source.pt"),
                        run_dir=Path("online"),
                        training_eval_data_selection=Path("eval/data-selection.json"),
                        generator_gate_worker_processes=0,
                    )
                )

            run.assert_not_called()

        run.assert_not_called()

    def test_generation_command_streams_stdout_and_returns_summary(self) -> None:
        class FakeProcess:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.stdout = iter(('{"game_count": ', "2}\n"))

            def wait(self) -> int:
                return 0

        with patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.Popen", FakeProcess):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                completed = _run_generation_command(["generate"], cwd=Path("."), env={})

        self.assertEqual(completed.stdout, '{"game_count": 2}\n')
        self.assertEqual(stdout.getvalue(), '{"game_count": 2}\n')

    def test_generation_command_failure_keeps_streamed_stdout(self) -> None:
        class FakeProcess:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.stdout = iter(("progress\n", "partial summary\n"))

            def wait(self) -> int:
                return 3

        with patch("intrep.problems.shogi_policy_value.generated_game_production.subprocess.Popen", FakeProcess):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    _run_generation_command(["generate"], cwd=Path("."), env={})

        self.assertEqual(raised.exception.returncode, 3)
        self.assertEqual(raised.exception.output, "progress\npartial summary\n")
        self.assertEqual(stdout.getvalue(), "progress\npartial summary\n")

    def test_online_replay_appends_generated_examples_and_trains_from_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
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
                            _mcts_record(("7g7f", "3c3d"), "black"),
                            _mcts_record(("2g2f", "8c8d"), "white"),
                        ],
                    )
                    return None
                if any(item.endswith("evaluate_shogi_players.py") for item in command):
                    gate_commands.append(command)
                    out_path = Path(command[command.index("--out") + 1])
                    player_a = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-a-checkpoint-id") + 1],
                        settings={},
                    )
                    player_b = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-b-checkpoint-id") + 1],
                        settings={},
                    )
                    write_shogi_game_records_jsonl(
                        out_path,
                        [
                            _mcts_record_with_actors(
                                ("7g7f", "3c3d"),
                                "black",
                                black_actor=player_a,
                                white_actor=player_b,
                            ),
                            _mcts_record_with_actors(
                                ("2g2f", "8c8d"),
                                "white",
                                black_actor=player_b,
                                white_actor=player_a,
                            ),
                        ],
                    )
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
                self.assertIsInstance(examples[0], LegalMovePolicyValueTensorSample)
                self.assertIsInstance(eval_examples[0], LegalMovePolicyValueTensorSample)
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
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
            self.assertEqual(train_batches, [4, 4])
            self.assertEqual(result.iterations[0].appended_examples, 4)
            self.assertEqual(result.iterations[0].replay_size, 4)
            self.assertEqual(result.iterations[0].sampled_examples, 4)
            self.assertEqual(result.stop_reason, None)
            self.assertEqual(len(gate_commands), 1)
            source_checkpoint_id = load_shogi_policy_value_checkpoint_identity(checkpoint_path).checkpoint_id
            self.assertEqual(gate_commands[0][gate_commands[0].index("--match-worker-processes") + 1], "4")
            self.assertEqual(
                gate_commands[0][gate_commands[0].index("--player-b-checkpoint-id") + 1],
                source_checkpoint_id,
            )
            gate_result = json.loads((run_dir / "iteration-0002" / "generator-gate-result.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_result["player_a_wins"], 2)
            self.assertEqual(gate_result["player_a_losses"], 0)
            self.assertEqual(gate_result["decision"], "favorable")
            self.assertFalse(gate_result["should_stop"])
            self.assertEqual(gate_result["margin"], 2)
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_black"],
                {"games": 1, "wins": 1, "losses": 0, "draws": 0, "unknown_results": 0},
            )
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_white"],
                {"games": 1, "wins": 1, "losses": 0, "draws": 0, "unknown_results": 0},
            )
            metrics = json.loads(result.iterations[0].metrics.read_text(encoding="utf-8"))
            self.assertEqual(metrics["checkpoint"]["init_id"], source_checkpoint_id)
            self.assertEqual(metrics["checkpoint"]["id"], load_shogi_policy_value_checkpoint_identity(result.iterations[0].checkpoint).checkpoint_id)
            self.assertEqual(
                metrics["checkpoint"]["best_id"],
                load_shogi_policy_value_checkpoint_identity(result.iterations[0].best_checkpoint).checkpoint_id,
            )
            self.assertEqual(metrics["replay"]["sampled_examples"], 4)
            self.assertEqual(metrics["replay"]["sampled_examples_per_iteration"], 3)
            self.assertEqual(metrics["replay"]["max_seed_examples_per_iteration"], 50000)
            self.assertEqual(metrics["training"]["batch_size"], 6)
            self.assertEqual(metrics["training"]["target_sample_passes"], 2.0)
            self.assertEqual(metrics["training"]["optimizer_steps_per_iteration"], 1)
            self.assertEqual(metrics["training"]["max_optimizer_steps_per_iteration"], 5)
            self.assertEqual(metrics["training"]["effective_sample_passes"], 1.5)
            self.assertIsNotNone(metrics["iteration"]["wall_time_sec"])
            self.assertIsNone(metrics["gate"]["wall_time_sec"])
            self.assertEqual(
                metrics["gate"]["config"],
                {
                    "games": 32,
                    "worker_processes": 4,
                    "mcts_simulations": 128,
                    "nn_leaf_eval_batch_limit": 64,
                    "max_plies": 320,
                },
            )
            self.assertIsNotNone(metrics["generation"]["wall_time_sec"])
            self.assertIsNotNone(metrics["generation"]["train_extraction_wall_time_sec"])
            self.assertIsNotNone(metrics["replay"]["generated_tensorize_wall_time_sec"])
            self.assertIsNotNone(metrics["replay"]["sampling_wall_time_sec"])
            self.assertIsNotNone(metrics["training"]["wall_time_sec"])
            self.assertIsNotNone(metrics["checkpoint"]["save_wall_time_sec"])

    def test_online_replay_continues_when_generator_gate_is_unclear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_mcts_record(("7g7f", "3c3d"), "black")])
                    return None
                if any(item.endswith("evaluate_shogi_players.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    player_a = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-a-checkpoint-id") + 1],
                        settings={},
                    )
                    player_b = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-b-checkpoint-id") + 1],
                        settings={},
                    )
                    write_shogi_game_records_jsonl(
                        out_path,
                        [
                            _mcts_record_with_actors(
                                ("7g7f", "3c3d"),
                                "black",
                                black_actor=player_a,
                                white_actor=player_b,
                            ),
                            _mcts_record_with_actors(
                                ("2g2f", "8c8d"),
                                "black",
                                black_actor=player_b,
                                white_actor=player_a,
                            ),
                            _mcts_record_with_actors(
                                ("7g7f", "8c8d"),
                                "white",
                                black_actor=player_a,
                                white_actor=player_b,
                            ),
                        ],
                    )
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": 3,
                                "player_a_wins": 1,
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
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run) as generation_run,
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", return_value=_training_result(ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, ))),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=2,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=2),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                    )
                )

            self.assertEqual(len(result.iterations), 2)
            self.assertEqual(result.stop_reason, None)
            self.assertEqual(generation_run.call_count, 2)
            gate_result = json.loads((run_dir / "iteration-0002" / "generator-gate-result.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_result["decision"], "unclear")
            self.assertFalse(gate_result["should_stop"])
            self.assertEqual(gate_result["margin"], -1)
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_black"],
                {"games": 2, "wins": 1, "losses": 1, "draws": 0, "unknown_results": 0},
            )
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_white"],
                {"games": 1, "wins": 0, "losses": 1, "draws": 0, "unknown_results": 0},
            )

    def test_online_replay_stops_when_generator_candidate_is_clearly_worse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_mcts_record(("7g7f", "3c3d"), "black")])
                    return None
                if any(item.endswith("evaluate_shogi_players.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    player_a = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-a-checkpoint-id") + 1],
                        settings={},
                    )
                    player_b = ShogiActorSpec(
                        kind="checkpoint",
                        name=command[command.index("--player-b-checkpoint-id") + 1],
                        settings={},
                    )
                    write_shogi_game_records_jsonl(
                        out_path,
                        [
                            _mcts_record_with_actors(
                                ("7g7f", "3c3d"),
                                "white",
                                black_actor=player_a,
                                white_actor=player_b,
                            ),
                            _mcts_record_with_actors(
                                ("2g2f", "8c8d"),
                                "black",
                                black_actor=player_b,
                                white_actor=player_a,
                            ),
                        ],
                    )
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
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run) as generation_run,
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", return_value=_training_result(ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, ))),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
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
            self.assertEqual(result.stop_reason, "generator_candidate_clearly_worse")
            self.assertEqual(result.stopped_iteration_index, 2)
            self.assertEqual(generation_run.call_count, 1)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((run_dir / "iteration-0002" / "generated-games.jsonl").exists())
            gate_result = json.loads((run_dir / "iteration-0002" / "generator-gate-result.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_result["decision"], "clearly_worse")
            self.assertTrue(gate_result["should_stop"])
            self.assertEqual(gate_result["margin"], -2)
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_black"],
                {"games": 1, "wins": 0, "losses": 1, "draws": 0, "unknown_results": 0},
            )
            self.assertEqual(
                gate_result["side_breakdown"]["player_a_as_white"],
                {"games": 1, "wins": 0, "losses": 1, "draws": 0, "unknown_results": 0},
            )

    def test_online_replay_seeds_replay_from_bundle_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            bundle_dir = root / "bundle"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            seed_train_record = _record(("9g9f", "9c9d"), "black")
            seed_eval_record = _record(("1g1f", "1c1d"), "white")
            generated_records = [
                _mcts_record(("7g7f", "3c3d"), "black"),
                _mcts_record(("2g2f", "8c8d"), "white"),
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
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=8),
                        replay_seed_data_selection=bundle_dir / "data-selection.json",
                        training_eval_data_selection=bundle_dir / "data-selection.json",
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=2),),
                    )
                )

            self.assertEqual(result.preloaded_examples, 0)
            self.assertEqual(result.training_eval_examples, 2)
            self.assertEqual(result.replay_seed_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(result.training_eval_data_selection, bundle_dir / "data-selection.json")
            self.assertEqual(train_batches, [6])
            self.assertEqual(eval_batches, [2])
            self.assertEqual(result.iterations[0].replay_size, 6)
            metrics = json.loads((run_dir / "iteration-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["replay"]["preloaded_examples"], 0)
            self.assertEqual(metrics["replay"]["seed_eligible_examples"], 2)
            self.assertEqual(metrics["replay"]["seed_sampled_examples"], 2)
            self.assertEqual(metrics["replay"]["generated_sampled_examples"], 4)
            self.assertEqual(metrics["replay"]["generated_replay_size"], 4)
            self.assertEqual(metrics["training"]["eval_examples"], 2)
            self.assertEqual(metrics["generation"]["generated_holdout_examples"], 0)
            self.assertEqual(metrics["training"]["eval_source"], "fixed_data_selection")
            self.assertEqual(metrics["replay"]["seed_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["training"]["eval_data_selection"], str(bundle_dir / "data-selection.json"))
            self.assertEqual(metrics["generation"]["summary"]["game_count"], 2)
            self.assertTrue((run_dir / "iteration-0001" / "generation-summary.json").exists())

    def test_online_replay_generates_multiple_experience_sources_per_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    if command[command.index("--white-kind") + 1] == "usi_engine":
                        records = [_mcts_record(("2g2f", "8c8d"), "white")]
                    elif command[command.index("--black-kind") + 1] == "usi_engine":
                        records = [_mcts_record(("7g7f", "3c3d"), "black")]
                    else:
                        records = [replace(_mcts_record(("7g7f", "3c3d"), None), end_reason="max_plies")]
                    write_shogi_game_records_jsonl(out_path, records)
                    end_reason = records[0].end_reason
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": len(records),
                                "average_plies": 2,
                                "end_reasons": {end_reason: len(records)},
                                "black_wins": 1 if records[0].winner == "black" else 0,
                                "white_wins": 1 if records[0].winner == "white" else 0,
                                "draws": 1 if records[0].winner is None else 0,
                                "generation_wall_time_sec": 1.0,
                                "plies_per_sec": 2.0,
                            }
                        )
                        + "\n",
                    )
                return None

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", return_value=_training_result(ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, ))),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
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
            self.assertEqual(summary["max_plies_draw_count"], 1)
            self.assertEqual(summary["max_plies_draw_rate"], 1 / 3)
            self.assertEqual(summary["game_over_count"], 2)
            self.assertEqual(summary["game_over_rate"], 2 / 3)
            self.assertEqual([source["name"] for source in summary["sources"]], ["self-play", "checkpoint-vs-usi", "usi-vs-checkpoint"])
            self.assertEqual(summary["sources"][1]["black_player"]["kind"], "checkpoint")
            self.assertEqual(summary["sources"][1]["white_player"]["kind"], "usi_engine")
            self.assertEqual(summary["sources"][2]["black_player"]["kind"], "usi_engine")
            self.assertEqual(summary["sources"][2]["white_player"]["kind"], "checkpoint")

    def test_online_replay_passes_training_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
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
                            _mcts_record(("7g7f", "3c3d"), "black"),
                            _mcts_record(("2g2f", "8c8d"), "white"),
                        ],
                    )

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                nonlocal captured_config
                captured_config = config
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
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
                        training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
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
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")

            def fake_run(command: list[str], **_kwargs: object) -> None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_mcts_record(("7g7f", "3c3d"), "black")])

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
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
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
                        training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, progress_every=2),
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
            _write_checkpoint(checkpoint_path)
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
                            _mcts_record(("7g7f", "3c3d"), "black"),
                            _mcts_record(("2g2f", "8c8d"), "white"),
                        ],
                    )

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_run) as run,
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
            ):
                result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
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

            self.assertEqual(train_batches, [8])
            self.assertTrue(result.iterations[0].training_skipped)
            self.assertEqual(result.iterations[0].sampled_examples, 0)
            self.assertEqual(result.iterations[0].best_checkpoint, checkpoint_path)
            self.assertFalse(result.iterations[1].training_skipped)
            first_metrics = json.loads((run_dir / "iteration-0001" / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(first_metrics["training"]["skipped"])
            self.assertEqual(first_metrics["training"]["skip_reason"], "min_replay_size")
            self.assertIsNone(first_metrics["training"]["effective_sample_passes"])
            self.assertIsNone(first_metrics["replay"]["sampling_wall_time_sec"])
            self.assertIsNone(first_metrics["training"]["wall_time_sec"])
            self.assertIsNone(first_metrics["checkpoint"]["save_wall_time_sec"])
            self.assertEqual(
                first_metrics["checkpoint"]["id"],
                load_shogi_policy_value_checkpoint_identity(checkpoint_path).checkpoint_id,
            )

    def test_online_replay_resume_reconstructs_generated_replay_from_completed_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "source.pt"
            _write_checkpoint(checkpoint_path)
            run_dir = root / "online"
            arena_repo = root / "arena"
            arena_repo.mkdir()
            training_eval_data_selection = _write_training_eval_bundle(root / "training-eval")
            train_batches: list[int] = []

            def fake_generation(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str] | None:
                if any(item.endswith("generate_shogi_games.py") for item in command):
                    out_path = Path(command[command.index("--out") + 1])
                    write_shogi_game_records_jsonl(out_path, [_mcts_record(("7g7f", "3c3d"), "black")])
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "game_count": 1,
                                "average_plies": 2,
                                "end_reasons": {"game_over": 1},
                                "black_wins": 1,
                                "white_wins": 0,
                                "draws": 0,
                                "generation_wall_time_sec": 1.0,
                                "plies_per_sec": 2.0,
                            }
                        )
                        + "\n",
                    )
                return None

            def fake_gate(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                out_path = Path(command[command.index("--out") + 1])
                player_a = command[command.index("--player-a-checkpoint-id") + 1]
                player_b = command[command.index("--player-b-checkpoint-id") + 1]
                write_shogi_game_records_jsonl(
                    out_path,
                    [
                        _mcts_record_with_actors(
                            ("7g7f", "3c3d"),
                            "black",
                            black_actor=ShogiActorSpec(kind="checkpoint", name=player_a, settings={}),
                            white_actor=ShogiActorSpec(kind="checkpoint", name=player_b, settings={}),
                        )
                    ],
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "game_count": 1,
                            "player_a_wins": 1,
                            "player_a_losses": 0,
                            "draws": 0,
                            "average_plies": 2,
                            "illegal_move_count": 0,
                        }
                    )
                    + "\n",
                )

            def fake_train(examples, *, eval_examples, config, initial_state_dict, progress_callback=None):
                train_batches.append(len(examples))
                return _training_result(config)

            with (
                patch("intrep.problems.shogi_policy_value.generated_game_production._run_generation_command", side_effect=fake_generation),
                patch("intrep.problems.shogi_policy_value.online_replay.subprocess.run", side_effect=fake_gate),
                patch("intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_state_dict", return_value={}),
                patch(
                    "intrep.problems.shogi_policy_value.online_replay.load_shogi_policy_value_checkpoint_training_config",
                    return_value=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2),
                ),
                patch("intrep.problems.shogi_policy_value.online_replay.train_shogi_policy_value_model", side_effect=fake_train),
            ):
                first_result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=1,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=8),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                    )
                )
                resumed_result = run_shogi_online_replay(
                    ShogiOnlineReplayConfig(training_config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
                        checkpoint=checkpoint_path,
                        run_dir=run_dir,
                        iterations=2,
                        resume=True,
                        replay_capacity=8,
                        min_replay_size=1,
                        training_budget=ShogiOnlineReplayTrainingBudget(sampled_examples_per_iteration=8),
                        training_eval_data_selection=training_eval_data_selection,
                        arena_repo=arena_repo,
                        experience_sources=(_self_play_source(games=1),),
                    )
                )

            self.assertEqual(train_batches, [2, 4])
            self.assertEqual(len(first_result.iterations), 1)
            self.assertEqual(len(resumed_result.iterations), 2)
            self.assertEqual(resumed_result.iterations[1].iteration_index, 2)
            self.assertFalse((run_dir / "iteration-0001" / "generated-train-games.jsonl").exists())
            self.assertFalse((run_dir / "iteration-0002" / "generated-train-games.jsonl").exists())
            second_metrics = json.loads((run_dir / "iteration-0002" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(second_metrics["replay"]["generated_replay_size"], 4)
            self.assertEqual(second_metrics["replay"]["generated_sampled_examples"], 4)


def _training_result(config: ShogiPolicyValueTrainingConfig) -> ShogiPolicyValueTrainingResult:
    return ShogiPolicyValueTrainingResult(
        model=build_shogi_policy_value_model(
            ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
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


def _write_checkpoint(path: Path) -> None:
    config = ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID, embedding_dim=8, hidden_dim=16, num_heads=2)
    save_shogi_policy_value_state_checkpoint(path, build_shogi_policy_value_model(config).state_dict(), config)


if __name__ == "__main__":
    unittest.main()
