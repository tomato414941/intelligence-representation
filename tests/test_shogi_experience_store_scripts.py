from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import shogi

from intrep.problems.shogi_policy_value.data_selection import load_shogi_policy_value_data_selection
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis, write_shogi_engine_analysis_jsonl
from intrep.worlds.shogi.experience_store import append_shogi_experience_store
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.game_trace import trace_shogi_game_record
from intrep.worlds.shogi.training_data_bundle import create_shogi_training_data_bundle


BLACK_ACTOR = ShogiActorSpec(
    kind="checkpoint",
    name="black-model",
    settings={"checkpoint": "runs/shogi/model-a/checkpoint.pt", "policy": "mcts", "simulations": 8},
)
WHITE_ACTOR = ShogiActorSpec(kind="usi_engine", name="white-engine", settings={"go_command": "go nodes 1"})
USI_ENGINE_ACTOR = ShogiActorSpec(kind="usi_engine", name="yaneuraou", settings={"go_command": "go nodes 1"})


def _record(
    moves: tuple[str, ...],
    winner: str | None,
    *,
    black_actor: ShogiActorSpec = BLACK_ACTOR,
    white_actor: ShogiActorSpec = WHITE_ACTOR,
) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=black_actor,
        white_actor=white_actor,
        winner=winner,
        end_reason="game_over",
    )


def _heldout_position_record() -> ShogiGameRecord:
    board = shogi.Board()
    board.push_usi("7g7f")
    return shogi_game_record_from_usi_moves(
        ("3c3d",),
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=board.sfen(),
        winner="black",
        end_reason="game_over",
    )


class ShogiExperienceStoreScriptsTest(unittest.TestCase):
    def test_appends_records_to_mutable_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "run-1.jsonl"
            second_input = root / "run-2.jsonl"
            store_dir = root / "data" / "shogi" / "experiences" / "main"
            first_records = [_record(("7g7f", "3c3d"), "black"), _record(("2g2f", "8c8d"), "white")]
            second_records = [_record(("5g5f", "5c5d"), "black")]
            write_shogi_game_records_jsonl(first_input, first_records)
            write_shogi_game_records_jsonl(second_input, second_records)

            first_result = append_shogi_experience_store(input_path=first_input, store_dir=store_dir)
            second_result = append_shogi_experience_store(input_path=second_input, store_dir=store_dir)

            self.assertEqual(first_result["added_games"], 2)
            self.assertEqual(second_result["added_games"], 1)
            self.assertEqual(load_shogi_game_records_jsonl(store_dir / "games.jsonl"), first_records + second_records)
            manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "intrep.shogi_experience_store.v1")
            self.assertEqual(manifest["game_count"], 3)
            self.assertEqual(manifest["position_stats"]["transition_count"], 6)
            self.assertEqual(manifest["position_stats"]["unique_position_count"], 4)
            self.assertEqual(manifest["position_stats"]["duplicate_position_count"], 2)
            self.assertEqual(manifest["position_stats"]["max_position_repeat_count"], 3)
            self.assertEqual(manifest["actor_pair_counts"], {"checkpoint:usi_engine": 3})
            self.assertEqual(
                manifest["checkpoint_actor_counts"],
                {"runs/shogi/model-a/checkpoint.pt | policy=mcts | simulations=8": 3},
            )
            self.assertEqual(
                manifest["checkpoint_actor_summaries"],
                [
                    {
                        "count": 3,
                        "actor_names": ["black-model"],
                        "checkpoint_id": "black-model",
                        "checkpoint_path": "runs/shogi/model-a/checkpoint.pt",
                        "move_selector": "mcts",
                        "mcts_simulations_per_move": 8,
                    }
                ],
            )
            history_lines = (store_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(history_lines), 2)
            second_history = json.loads(history_lines[1])
            self.assertEqual(second_history["added_actor_pair_counts"], {"checkpoint:usi_engine": 1})
            self.assertEqual(
                second_history["added_checkpoint_actor_counts"],
                {"runs/shogi/model-a/checkpoint.pt | policy=mcts | simulations=8": 1},
            )
            self.assertEqual(second_history["added_checkpoint_actor_summaries"][0]["count"], 1)
            self.assertEqual(second_history["total_checkpoint_actor_summaries"][0]["count"], 3)
            self.assertEqual(second_history["total_games"], 3)
            self.assertEqual(second_history["total_position_stats"]["unique_position_count"], 4)

    def test_append_experience_store_script_is_thin_cli_wrapper(self) -> None:
        append_module = _load_script_module("append_shogi_experience_store")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "games.jsonl"
            store_dir = root / "store"
            write_shogi_game_records_jsonl(input_path, [_record(("7g7f", "3c3d"), "black")])

            with patch("sys.stdout", new_callable=StringIO):
                append_module.main(["--input", str(input_path), "--store", str(store_dir)])

            self.assertTrue((store_dir / "games.jsonl").exists())
            self.assertTrue((store_dir / "manifest.json").exists())

    def test_checkpoint_actor_summary_groups_same_checkpoint_settings_across_sides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "games.jsonl"
            store_dir = root / "store"
            black_actor = ShogiActorSpec(
                kind="checkpoint",
                name="black",
                settings={
                    "checkpoint_id": "model-a",
                    "checkpoint_path": "models/model-a/checkpoint.pt",
                    "move_selector": "mcts",
                    "mcts_simulations_per_move": 16,
                },
            )
            white_actor = ShogiActorSpec(
                kind="checkpoint",
                name="white",
                settings={
                    "checkpoint_id": "model-a",
                    "checkpoint_path": "models/model-a/checkpoint.pt",
                    "move_selector": "mcts",
                    "mcts_simulations_per_move": 16,
                },
            )
            write_shogi_game_records_jsonl(
                input_path,
                [_record(("7g7f", "3c3d"), "black", black_actor=black_actor, white_actor=white_actor)],
            )

            append_shogi_experience_store(input_path=input_path, store_dir=store_dir)

            manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["checkpoint_actor_summaries"],
                [
                    {
                        "count": 2,
                        "actor_names": ["black", "white"],
                        "checkpoint_id": "model-a",
                        "checkpoint_path": "models/model-a/checkpoint.pt",
                        "move_selector": "mcts",
                        "mcts_simulations_per_move": 16,
                    }
                ],
            )

    def test_creates_fixed_training_data_bundle_from_explicit_train_eval_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-source.jsonl"
            eval_path = root / "eval-source.jsonl"
            output_root = root / "data" / "shogi" / "datasets"
            train_records = [
                _record(("7g7f", "3c3d"), "black"),
                _record(("5g5f", "5c5d"), "black", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
            ]
            eval_records = [
                _record(("2g2f", "8c8d"), "white"),
                _record(("6g6f", "6c6d"), "white", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
            ]
            write_shogi_game_records_jsonl(train_path, train_records)
            write_shogi_game_records_jsonl(eval_path, eval_records)

            result = create_shogi_training_data_bundle(
                train_games=train_path,
                eval_games=eval_path,
                name="main-view-0001",
                output_root=output_root,
                max_train_games=1,
                max_eval_games=1,
            )

            view_dir = output_root / "main-view-0001"
            self.assertEqual(result["training_data_bundle"], str(view_dir))
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "games.jsonl"), [train_records[0], eval_records[0]])
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "train-games.jsonl"), [train_records[0]])
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "eval-games.jsonl"), [eval_records[0]])
            definition = load_shogi_policy_value_data_selection(view_dir / "data-selection.json")
            self.assertEqual(definition.name, "main-view-0001")
            self.assertEqual(definition.target_construction.policy, "chosen_move")
            self.assertEqual(definition.target_construction.policy_temperature_cp, 100.0)
            self.assertEqual(definition.target_construction.policy_mate_cp, 100000.0)
            self.assertEqual(definition.target_construction.value, "winner")
            self.assertEqual(definition.target_construction.score_cp_scale, 600.0)
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-games.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-games.jsonl")
            self.assertEqual(definition.train_sources[0].max_games, 1)
            self.assertEqual(definition.eval_sources[0].max_games, 1)
            manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "intrep.shogi_training_data_bundle.v1")
            self.assertEqual(manifest["train_source_games_jsonl"], [str(train_path)])
            self.assertEqual(manifest["eval_source_games_jsonl"], str(eval_path))
            self.assertEqual(manifest["max_train_games"], 1)
            self.assertEqual(manifest["max_eval_games"], 1)
            self.assertEqual(manifest["eval_position_policy"], "allow_overlap")
            self.assertEqual(manifest["selected_eval_games_before_position_policy"], 1)
            self.assertEqual(manifest["skipped_eval_games_for_train_position_overlap"], 0)
            self.assertEqual(manifest["actor_pair_counts"], {"checkpoint:usi_engine": 2})
            self.assertEqual(manifest["train_actor_pair_counts"], {"checkpoint:usi_engine": 1})
            self.assertEqual(manifest["eval_actor_pair_counts"], {"checkpoint:usi_engine": 1})
            self.assertEqual(manifest["checkpoint_actor_summaries"][0]["count"], 2)
            self.assertEqual(manifest["train_checkpoint_actor_summaries"][0]["count"], 1)
            self.assertEqual(manifest["eval_checkpoint_actor_summaries"][0]["count"], 1)
            self.assertEqual(
                manifest["checkpoint_actor_summaries"][0]["checkpoint_path"],
                "runs/shogi/model-a/checkpoint.pt",
            )
            self.assertEqual(manifest["train_games"], 1)
            self.assertEqual(manifest["eval_games"], 1)
            self.assertEqual(manifest["train_position_stats"]["transition_count"], 2)
            self.assertEqual(manifest["eval_position_stats"]["transition_count"], 2)
            self.assertEqual(manifest["train_eval_position_overlap_count"], 1)
            self.assertEqual(manifest["position_stats"]["unique_position_count"], 3)
            self.assertEqual(manifest["target_construction"]["policy"], "chosen_move")
            self.assertEqual(manifest["target_construction"]["value"], "winner")

    def test_training_data_bundle_can_exclude_eval_games_with_train_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-source.jsonl"
            eval_path = root / "eval-source.jsonl"
            output_root = root / "datasets"
            train_record = _record(("7g7f",), "black")
            overlapping_eval_record = _record(("2g2f",), "white")
            heldout_eval_record = _heldout_position_record()
            write_shogi_game_records_jsonl(train_path, [train_record])
            write_shogi_game_records_jsonl(eval_path, [overlapping_eval_record, heldout_eval_record])

            create_shogi_training_data_bundle(
                train_games=train_path,
                eval_games=eval_path,
                name="heldout-eval",
                output_root=output_root,
                eval_position_policy="exclude_train_position_games",
            )

            bundle_dir = output_root / "heldout-eval"
            self.assertEqual(load_shogi_game_records_jsonl(bundle_dir / "eval-games.jsonl"), [heldout_eval_record])
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eval_position_policy"], "exclude_train_position_games")
            self.assertEqual(manifest["selected_eval_games_before_position_policy"], 2)
            self.assertEqual(manifest["skipped_eval_games_for_train_position_overlap"], 1)
            self.assertEqual(manifest["train_eval_position_overlap_count"], 0)
            self.assertEqual(manifest["train_eval_position_overlap_ratio"], 0.0)

    def test_training_data_bundle_can_include_engine_analysis_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-source.jsonl"
            eval_path = root / "eval-source.jsonl"
            analysis_path = root / "analysis-source.jsonl"
            output_root = root / "datasets"
            train_record = _record(("7g7f",), "black")
            eval_record = _record(("2g2f",), "black")
            train_transition = trace_shogi_game_record(train_record).transitions[0]
            write_shogi_game_records_jsonl(train_path, [train_record])
            write_shogi_game_records_jsonl(eval_path, [eval_record])
            write_shogi_engine_analysis_jsonl(
                analysis_path,
                [
                    ShogiEngineAnalysis(
                        position_sfen=train_transition.position_sfen,
                        legal_moves=train_transition.legal_moves,
                        engine=USI_ENGINE_ACTOR,
                        usi_info_lines=("info multipv 1 score cp 100 pv 7g7f",),
                    )
                ],
            )

            result = create_shogi_training_data_bundle(
                train_games=train_path,
                eval_games=eval_path,
                analysis_sources=(analysis_path,),
                name="engine-analysis-bundle",
                output_root=output_root,
                policy_target_construction="engine_analysis_multipv",
                value_target_construction="engine_analysis_score",
            )

            bundle_dir = output_root / "engine-analysis-bundle"
            copied_analysis_path = bundle_dir / "analysis.jsonl"
            self.assertEqual(result["analysis_jsonl"], [str(copied_analysis_path)])
            self.assertTrue(copied_analysis_path.exists())
            definition = load_shogi_policy_value_data_selection(bundle_dir / "data-selection.json")
            self.assertEqual(definition.analysis_sources[0].path, copied_analysis_path)
            self.assertEqual(definition.target_construction.policy, "engine_analysis_multipv")
            self.assertEqual(definition.target_construction.value, "engine_analysis_score")
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["analysis_source_jsonl"], [str(analysis_path)])
            self.assertEqual(manifest["files"]["analysis"], ["analysis.jsonl"])
            self.assertEqual(manifest["analysis_coverage"]["train"]["positions"], 1)
            self.assertEqual(manifest["analysis_coverage"]["train"]["covered"], 1)
            self.assertEqual(manifest["analysis_coverage"]["train"]["ratio"], 1.0)
            self.assertEqual(manifest["analysis_coverage"]["eval"]["positions"], 1)
            self.assertEqual(manifest["analysis_coverage"]["eval"]["covered"], 1)
            self.assertEqual(manifest["analysis_coverage"]["eval"]["ratio"], 1.0)

    def test_training_data_bundle_rejects_engine_analysis_targets_without_analysis_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-source.jsonl"
            eval_path = root / "eval-source.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(train_path, [_record(("7g7f",), "black")])
            write_shogi_game_records_jsonl(eval_path, [_record(("2g2f",), "black")])

            with self.assertRaisesRegex(ValueError, "analysis_sources"):
                create_shogi_training_data_bundle(
                    train_games=train_path,
                    eval_games=eval_path,
                    name="missing-analysis",
                    output_root=output_root,
                    policy_target_construction="engine_analysis_multipv",
                    value_target_construction="engine_analysis_score",
                )

    def test_training_data_bundle_script_is_thin_cli_wrapper(self) -> None:
        view_module = _load_script_module("create_shogi_training_data_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(train_path, [_record(("7g7f", "3c3d"), "black")])
            write_shogi_game_records_jsonl(eval_path, [_record(("2g2f", "8c8d"), "white")])

            with patch("sys.stdout", new_callable=StringIO):
                view_module.main(
                    [
                        "--train-games",
                        str(train_path),
                        "--eval-games",
                        str(eval_path),
                        "--name",
                        "cli-view",
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertTrue((output_root / "cli-view" / "data-selection.json").exists())

    def test_training_data_bundle_script_warns_on_multiple_train_inputs(self) -> None:
        view_module = _load_script_module("create_shogi_training_data_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_a = root / "train-a.jsonl"
            train_b = root / "train-b.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(train_a, [_record(("7g7f", "3c3d"), "black")])
            write_shogi_game_records_jsonl(train_b, [_record(("2g2f", "8c8d"), "white")])
            write_shogi_game_records_jsonl(eval_path, [_record(("5g5f", "5c5d"), "black")])

            stderr = StringIO()
            with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", stderr):
                view_module.main(
                    [
                        "--train-games",
                        str(train_a),
                        "--train-games",
                        str(train_b),
                        "--eval-games",
                        str(eval_path),
                        "--name",
                        "cli-view-multi-train",
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertIn("multiple --train-games inputs", stderr.getvalue())
            self.assertTrue((output_root / "cli-view-multi-train" / "data-selection.json").exists())

    def test_creates_training_data_bundle_from_selected_game_record_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_a = root / "train-a.jsonl"
            train_b = root / "train-b.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            checkpoint_records = [
                _record(("7g7f", "3c3d"), "black"),
                _record(("2g2f", "8c8d"), "white"),
                _record(("5g5f", "5c5d"), "black"),
            ]
            usi_engine_records = [
                _record(("6g6f", "6c6d"), "white", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
                _record(("4g4f", "4c4d"), "black", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
                _record(("3g3f", "3c3d"), "white", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
            ]
            eval_records = [
                _record(("8g8f", "8c8d"), "black"),
                _record(("9g9f", "9c9d"), "white"),
            ]
            write_shogi_game_records_jsonl(train_a, checkpoint_records)
            write_shogi_game_records_jsonl(train_b, usi_engine_records)
            write_shogi_game_records_jsonl(eval_path, eval_records)

            result = create_shogi_training_data_bundle(
                train_games=(train_a, train_b),
                eval_games=eval_path,
                name="main-view-selected",
                output_root=output_root,
                max_train_games=4,
                max_eval_games=1,
                actor_pair_ratios={"checkpoint:usi_engine": 0.5, "usi_engine:usi_engine": 0.5},
                seed=11,
            )

            view_dir = output_root / "main-view-selected"
            self.assertEqual(result["training_data_bundle"], str(view_dir))
            self.assertEqual(result["train_games"], 4)
            self.assertEqual(result["eval_games"], 1)
            train_records = load_shogi_game_records_jsonl(view_dir / "train-games.jsonl")
            self.assertEqual(len(train_records), 4)
            definition = load_shogi_policy_value_data_selection(view_dir / "data-selection.json")
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-games.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-games.jsonl")
            manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "intrep.shogi_training_data_bundle.v1")
            self.assertEqual(manifest["train_source_games_jsonl"], [str(train_a), str(train_b)])
            self.assertEqual(manifest["available_train_games"], 6)
            self.assertEqual(manifest["available_eval_games"], 2)
            self.assertEqual(manifest["actor_pair_ratios"], {"checkpoint:usi_engine": 0.5, "usi_engine:usi_engine": 0.5})
            self.assertEqual(manifest["train_actor_pair_counts"], {"checkpoint:usi_engine": 2, "usi_engine:usi_engine": 2})
            self.assertEqual(manifest["eval_actor_pair_counts"], {"checkpoint:usi_engine": 1})

    def test_training_data_bundle_script_selects_from_input_games(self) -> None:
        view_module = _load_script_module("create_shogi_training_data_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(
                train_path,
                [
                    _record(("7g7f", "3c3d"), "black"),
                    _record(("2g2f", "8c8d"), "white"),
                ],
            )
            write_shogi_game_records_jsonl(eval_path, [_record(("5g5f", "5c5d"), "black")])

            with patch("sys.stdout", new_callable=StringIO):
                view_module.main(
                    [
                        "--train-games",
                        str(train_path),
                        "--eval-games",
                        str(eval_path),
                        "--name",
                        "cli-view-selected",
                        "--output-root",
                        str(output_root),
                        "--max-train-games",
                        "1",
                    ]
                )

            view_dir = output_root / "cli-view-selected"
            self.assertTrue((view_dir / "data-selection.json").exists())
            manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_games"], 1)

    def test_refuses_to_overwrite_existing_training_data_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(
                train_path,
                [_record(("7g7f", "3c3d"), "black")],
            )
            write_shogi_game_records_jsonl(
                eval_path,
                [_record(("2g2f", "8c8d"), "white")],
            )
            (output_root / "existing").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                create_shogi_training_data_bundle(
                    train_games=train_path,
                    eval_games=eval_path,
                    name="existing",
                    output_root=output_root,
                )


def _load_script_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
