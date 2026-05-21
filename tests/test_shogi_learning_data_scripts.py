from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import shogi

from intrep.problems.shogi_policy_value.data_selection import load_shogi_policy_value_data_selection
from intrep.problems.shogi_policy_value.examples import load_shogi_move_policy_value_examples_jsonl
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis, write_shogi_engine_analysis_jsonl
from intrep.worlds.shogi.generated_record_archive import archive_shogi_generated_records
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
    settings={
        "checkpoint": "runs/shogi/model-a/checkpoint.pt",
        "move_selector": "mcts",
        "mcts_simulations_per_move": 8,
    },
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


class ShogiLearningDataScriptsTest(unittest.TestCase):
    def test_archives_generated_records_as_durable_record_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "run" / "generated-games.jsonl"
            output_root = root / "data" / "shogi" / "records" / "generated"
            write_shogi_game_records_jsonl(
                input_path,
                [
                    _record(("7g7f", "3c3d"), "black"),
                    _record(("2g2f", "8c8d", "2f2e"), "white"),
                ],
            )

            result = archive_shogi_generated_records(
                input_path=input_path,
                output_root=output_root,
                record_set_id="online-replay-20260518-001",
                source_run="runs/shogi/online-replay-x",
                generation_method="online_replay",
            )

            archive_dir = output_root / "online-replay-20260518-001"
            self.assertEqual(result["game_count"], 2)
            self.assertEqual(load_shogi_game_records_jsonl(archive_dir / "games.jsonl"), load_shogi_game_records_jsonl(input_path))
            manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "intrep.shogi_generated_record_archive.v1")
            self.assertEqual(manifest["record_schema"], "shogi_game_record_jsonl")
            self.assertEqual(manifest["source_name"], "generated")
            self.assertEqual(manifest["source_path"], str(input_path))
            self.assertEqual(manifest["source_run"], "runs/shogi/online-replay-x")
            self.assertEqual(manifest["generation_method"], "online_replay")
            self.assertEqual(manifest["game_count"], 2)
            self.assertEqual(manifest["transition_count"], 5)
            self.assertEqual(manifest["files"], {"games": "games.jsonl"})

    def test_archive_generated_records_refuses_to_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "generated-games.jsonl"
            output_root = root / "records" / "generated"
            write_shogi_game_records_jsonl(input_path, [_record(("7g7f", "3c3d"), "black")])
            archive_shogi_generated_records(
                input_path=input_path,
                output_root=output_root,
                record_set_id="existing",
            )

            with self.assertRaises(FileExistsError):
                archive_shogi_generated_records(
                    input_path=input_path,
                    output_root=output_root,
                    record_set_id="existing",
                )

    def test_archive_generated_records_script_is_thin_cli_wrapper(self) -> None:
        archive_module = _load_script_module("archive_shogi_generated_records")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "generated-games.jsonl"
            output_root = root / "records" / "generated"
            write_shogi_game_records_jsonl(input_path, [_record(("7g7f", "3c3d"), "black")])

            with patch("sys.stdout", new_callable=StringIO):
                archive_module.main(
                    [
                        "--input",
                        str(input_path),
                        "--record-set-id",
                        "cli-archive",
                        "--output-root",
                        str(output_root),
                        "--source-run",
                        "runs/shogi/run-a",
                    ]
                )

            self.assertTrue((output_root / "cli-archive" / "games.jsonl").exists())
            manifest = json.loads((output_root / "cli-archive" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_run"], "runs/shogi/run-a")

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
            self.assertEqual(len(load_shogi_move_policy_value_examples_jsonl(view_dir / "train-examples.jsonl")), 2)
            self.assertEqual(len(load_shogi_move_policy_value_examples_jsonl(view_dir / "eval-examples.jsonl")), 2)
            definition = load_shogi_policy_value_data_selection(view_dir / "data-selection.json")
            self.assertEqual(definition.name, "main-view-0001")
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-examples.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-examples.jsonl")
            self.assertEqual(definition.train_sources[0].max_examples, None)
            self.assertEqual(definition.eval_sources[0].max_examples, None)
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
            self.assertEqual(len(load_shogi_move_policy_value_examples_jsonl(bundle_dir / "eval-examples.jsonl")), 1)
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
            self.assertEqual(definition.train_sources[0].path, bundle_dir / "train-examples.jsonl")
            train_examples = load_shogi_move_policy_value_examples_jsonl(bundle_dir / "train-examples.jsonl")
            self.assertEqual(train_examples[0].policy_targets, {"7g7f": 1.0})
            self.assertIsNotNone(train_examples[0].value_target)
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
            train_examples = load_shogi_move_policy_value_examples_jsonl(view_dir / "train-examples.jsonl")
            self.assertEqual(len(train_examples), 8)
            definition = load_shogi_policy_value_data_selection(view_dir / "data-selection.json")
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-examples.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-examples.jsonl")
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

    def test_modal_tensor_cache_builder_releases_remote_cache_to_local_path(self) -> None:
        modal_builder = _load_script_module("modal_build_shogi_policy_value_tensor_cache")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_cache = root / "bundle" / "cache" / "legal-move"
            local_cache.mkdir(parents=True)
            (local_cache / "stale.txt").write_text("old\n", encoding="utf-8")

            with patch.object(modal_builder.subprocess, "run") as run:
                result = modal_builder._release_remote_cache_to_local(
                    remote_bundle="qhapaq-full",
                    cache_name="legal-move",
                    local_cache=local_cache,
                )

            self.assertFalse((local_cache / "stale.txt").exists())
            self.assertEqual(result["local_cache"], str(local_cache))
            run.assert_called_once_with(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    modal_builder.VOLUME_NAME,
                    "/qhapaq-full/cache/legal-move",
                    str(local_cache.parent),
                ],
                check=True,
            )

    def test_modal_tensor_cache_builder_defaults_local_release_path_to_bundle_cache(self) -> None:
        modal_builder = _load_script_module("modal_build_shogi_policy_value_tensor_cache")
        with tempfile.TemporaryDirectory() as directory:
            local_bundle = Path(directory) / "bundle"

            self.assertEqual(
                modal_builder._local_cache_path(
                    local_bundle=local_bundle,
                    output_space="policy_plane",
                    local_cache=None,
                ),
                local_bundle / "cache" / "policy-plane",
            )

    def test_modal_tensor_cache_builder_upload_overwrites_remote_bundle_files(self) -> None:
        modal_builder = _load_script_module("modal_build_shogi_policy_value_tensor_cache")
        with tempfile.TemporaryDirectory() as directory:
            local_bundle = Path(directory) / "bundle"
            local_bundle.mkdir()
            upload_context = MagicMock()
            batch = upload_context.__enter__.return_value
            volume = MagicMock()
            volume.batch_upload.return_value = upload_context

            with patch.object(modal_builder, "volume", volume, create=True):
                modal_builder._upload_bundle(local_bundle=local_bundle, remote_bundle="qhapaq-full")

            volume.batch_upload.assert_called_once_with(force=True)
            batch.put_directory.assert_called_once_with(str(local_bundle), "/qhapaq-full")

    def test_modal_tensor_cache_builder_names_shard_manifest_path_from_task(self) -> None:
        modal_builder = _load_script_module("modal_build_shogi_policy_value_tensor_cache")

        self.assertEqual(
            modal_builder._task_shard_manifest_relative_path(
                {
                    "split": "train",
                    "source_index": 2,
                    "source_example_start_index": 10000,
                    "source_example_end_index": 20000,
                }
            ),
            "train/source-0002-examples-00010000-00020000.json",
        )

    def test_modal_tensor_cache_builder_reads_completed_shard_manifest_paths(self) -> None:
        modal_builder = _load_script_module("modal_build_shogi_policy_value_tensor_cache")
        file_entry = type("FileEntry", (), {})
        paths = [
            "qhapaq-full/cache/policy-plane/train/source-0000-examples-00000000-00010000.json",
            "qhapaq-full/cache/policy-plane/eval/source-0000-examples-00000000-00010000.json",
            "qhapaq-full/cache/policy-plane/manifest.json",
            "qhapaq-full/cache/policy-plane/train/source-0000-examples-00000000-00010000.pt",
        ]
        entries = []
        for path in paths:
            entry = file_entry()
            entry.path = path
            entries.append(entry)
        volume = MagicMock()
        volume.listdir.return_value = entries

        with patch.object(modal_builder, "volume", volume, create=True):
            completed = modal_builder._completed_shard_manifest_paths(
                remote_bundle="qhapaq-full",
                cache_name="policy-plane",
            )

        self.assertEqual(
            completed,
            {
                "train/source-0000-examples-00000000-00010000.json",
                "eval/source-0000-examples-00000000-00010000.json",
            },
        )
        volume.listdir.assert_called_once_with("/qhapaq-full/cache/policy-plane", recursive=True)


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
