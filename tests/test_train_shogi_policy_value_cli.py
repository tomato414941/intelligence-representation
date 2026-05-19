import tempfile
import unittest
import json
import shutil
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from intrep.problems.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint
from intrep.problems.shogi_policy_value.data_selection import (
    load_shogi_policy_value_data_selection,
    load_shogi_policy_value_data_selection_examples,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.examples import CompactPolicyPlaneValueTensorSample
from intrep.problems.shogi_policy_value.model import SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER
from intrep.problems.shogi_policy_value.output_space import (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
)
from intrep.problems.shogi_policy_value.tensor_cache import build_shogi_policy_value_tensor_cache
from intrep.problems.shogi_policy_value.tensor_cache import (
    build_shogi_policy_value_tensor_cache_shard,
    load_shogi_policy_value_tensor_cache,
    write_shogi_policy_value_tensor_cache_manifest,
)
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis, write_shogi_engine_analysis_jsonl
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    ShogiMoveRecord,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.game_trace import trace_shogi_game_record
from intrep.worlds.shogi.position_encoding import (
    SHOGI_POSITION_FEATURE_MANIFEST,
    SHOGI_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_POSITION_INPUT_SCHEMA_ID,
)
from intrep.train_shogi_policy_value import main


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        winner=winner,
    )


def _record_with_multipv_info(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    record = _record(moves, winner)
    return replace(
        record,
        moves=tuple(
            ShogiMoveRecord(
                action_usi=move.action_usi,
                decision_usi_info_lines=(f"info multipv 1 score cp 100 pv {move.action_usi}",),
            )
            for move in record.moves
        ),
    )


class TrainShogiPolicyValueCliTest(unittest.TestCase):
    def test_trains_from_game_records_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            checkpoint_path = root / "shogi.pt"
            best_checkpoint_path = root / "shogi-best.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--best-checkpoint-path",
                    str(best_checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--max-train-eval-examples",
                    "2",
                    "--max-eval-examples",
                    "2",
                    "--log-every",
                    "1",
                    "--eval-every",
                    "1",
                    "--early-stopping-patience",
                    "1",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO) as stdout:
                main()

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(best_checkpoint_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertIn("step=1/1", stdout.getvalue())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["data_selection"]["name"], "test-shogi-policy-value")
            self.assertEqual(metrics["best_checkpoint_path"], str(best_checkpoint_path))
            self.assertIn(metrics["metrics"]["best_eval_step"], {0, 1})
            self.assertIsNotNone(metrics["metrics"]["best_eval_loss"])
            self.assertEqual(metrics["config"]["num_workers"], 0)
            self.assertEqual(metrics["config"]["early_stopping_patience"], 1)
            self.assertEqual(metrics["config"]["policy_loss_weight"], 1.0)
            self.assertEqual(metrics["config"]["value_loss_weight"], 1.0)
            self.assertEqual(metrics["raw_train_case_count"], 2)
            self.assertEqual(metrics["raw_eval_case_count"], 2)
            self.assertEqual(metrics["used_eval_case_count"], 2)
            self.assertEqual(metrics["metrics"]["eval_case_count"], 2)
            self.assertIn("actual_steps", metrics["metrics"])
            self.assertIn("stopped_early", metrics["metrics"])
            self.assertEqual(metrics["train_policy_target_summary"]["available_count"], 0)
            self.assertEqual(metrics["train_policy_target_summary"]["missing_count"], 2)
            self.assertEqual(metrics["eval_policy_target_summary"]["available_ratio"], 0.0)

    def test_trains_policy_plane_model_from_policy_plane_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-plane-value-tensors"
            checkpoint_path = root / "shogi-policy-plane.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-plane-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                output_space=SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
                shard_games=1,
            )
            manifest = json.loads((tensor_cache_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["output_space"], SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE)
            self.assertIn("max_legal_action_count", manifest)
            self.assertIn("max_target_action_count", manifest)
            cache = load_shogi_policy_value_tensor_cache(
                tensor_cache_path,
                expected_output_space=SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
            )
            self.assertIsInstance(cache.train_samples[0], CompactPolicyPlaneValueTensorSample)

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--tensor-cache",
                    str(tensor_cache_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--model",
                    SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["config"]["model"], SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER)
            self.assertEqual(metrics["tensor_cache_path"], str(tensor_cache_path))
            self.assertEqual(metrics["tensor_cache_output_space"], SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE)

    def test_tensor_cache_rejects_output_space_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                output_space=SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE,
                shard_games=1,
            )

            with self.assertRaisesRegex(ValueError, "output_space"):
                load_shogi_policy_value_tensor_cache(
                    tensor_cache_path,
                    expected_output_space=SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
                )
    def test_writes_policy_target_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record_with_multipv_info(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record_with_multipv_info(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "decision_usi_multipv",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["train_policy_target_summary"]["available_count"], 2)
            self.assertEqual(metrics["train_policy_target_summary"]["missing_count"], 0)
            self.assertEqual(metrics["train_policy_target_summary"]["available_ratio"], 1.0)
            self.assertEqual(metrics["train_policy_target_summary"]["mean_nonzero_count"], 1.0)
            self.assertEqual(metrics["eval_policy_target_summary"]["available_count"], 2)
            self.assertEqual(metrics["eval_policy_target_summary"]["missing_count"], 0)

    def test_trains_from_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
            )
            self.assertTrue((tensor_cache_path / "manifest.json").exists())
            self.assertTrue((tensor_cache_path / "train").exists())
            self.assertTrue((tensor_cache_path / "eval").exists())
            self.assertTrue(list((tensor_cache_path / "train").glob("*.json")))
            manifest = json.loads((tensor_cache_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_schema_id"], SHOGI_POSITION_INPUT_SCHEMA_ID)

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--tensor-cache",
                    str(tensor_cache_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                ],
            ), patch("sys.stdout", new_callable=StringIO), patch(
                "intrep.train_shogi_policy_value.load_shogi_policy_value_data_selection_examples",
                side_effect=AssertionError("tensor cache training should not rebuild examples from JSONL"),
            ):
                main()

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["tensor_cache_path"], str(tensor_cache_path))
            self.assertEqual(metrics["tensor_cache_output_space"], SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE)
            self.assertEqual(metrics["raw_train_case_count"], 2)
            self.assertEqual(metrics["raw_eval_case_count"], 2)

    def test_tensor_cache_identity_survives_bundle_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            moved_root = root / "moved"
            source_root.mkdir()
            moved_root.mkdir()
            train_games_path = source_root / "train-games.jsonl"
            eval_games_path = source_root / "eval-games.jsonl"
            data_selection_path = source_root / "data-selection.json"
            tensor_cache_path = source_root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": "train-games.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
            )
            manifest_path = tensor_cache_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["data_selection"] = shogi_policy_value_data_selection_to_json(
                load_shogi_policy_value_data_selection(data_selection_path)
            )
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            shutil.copytree(tensor_cache_path, moved_root / "cache" / "shogi-policy-value-tensors")
            shutil.copy2(train_games_path, moved_root / "train-games.jsonl")
            shutil.copy2(eval_games_path, moved_root / "eval-games.jsonl")
            shutil.copy2(data_selection_path, moved_root / "data-selection.json")
            moved_selection_path = moved_root / "data-selection.json"
            moved_selection = load_shogi_policy_value_data_selection(moved_selection_path)

            cache = load_shogi_policy_value_tensor_cache(
                moved_root / "cache" / "shogi-policy-value-tensors",
                expected_data_selection=moved_selection,
                expected_data_selection_root=moved_selection_path.parent,
            )

            self.assertEqual(len(cache.train_samples), 2)
            self.assertEqual(len(cache.eval_samples), 2)

    def test_tensor_cache_rejects_missing_input_schema_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
            )
            manifest_path = tensor_cache_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("input_schema_id")
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "input schema"):
                load_shogi_policy_value_tensor_cache(tensor_cache_path)

    def test_tensor_cache_rejects_missing_input_feature_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
            )
            manifest_path = tensor_cache_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("input_feature_manifest_hash")
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "input feature manifest"):
                load_shogi_policy_value_tensor_cache(tensor_cache_path)

    def test_tensor_cache_build_can_resume_existing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            first = build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
            )
            shard_paths = sorted((tensor_cache_path / "train").glob("*.pt")) + sorted((tensor_cache_path / "eval").glob("*.pt"))
            mtimes = {path: path.stat().st_mtime_ns for path in shard_paths}
            second = build_shogi_policy_value_tensor_cache(
                data_selection_path=data_selection_path,
                output_path=tensor_cache_path,
                shard_games=1,
                resume=True,
            )

            self.assertEqual(second["train_count"], first["train_count"])
            self.assertEqual(second["eval_count"], first["eval_count"])
            self.assertEqual({path: path.stat().st_mtime_ns for path in shard_paths}, mtimes)

    def test_tensor_cache_manifest_can_be_written_from_independent_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            build_shogi_policy_value_tensor_cache_shard(
                data_selection_path=data_selection_path,
                cache_dir=tensor_cache_path,
                split="train",
                source_index=0,
                source_example_start_index=0,
                source_example_end_index=2,
                shard_index=0,
            )
            build_shogi_policy_value_tensor_cache_shard(
                data_selection_path=data_selection_path,
                cache_dir=tensor_cache_path,
                split="eval",
                source_index=0,
                source_example_start_index=0,
                source_example_end_index=2,
                shard_index=0,
            )
            manifest = write_shogi_policy_value_tensor_cache_manifest(
                data_selection_path=data_selection_path,
                cache_dir=tensor_cache_path,
                shard_games=1,
            )

            self.assertEqual(manifest["input_schema_id"], SHOGI_POSITION_INPUT_SCHEMA_ID)
            self.assertEqual(manifest["input_feature_manifest"], SHOGI_POSITION_FEATURE_MANIFEST)
            self.assertEqual(manifest["input_feature_manifest_hash"], SHOGI_POSITION_FEATURE_MANIFEST_HASH)
            self.assertEqual(manifest["train_count"], 2)
            self.assertEqual(manifest["eval_count"], 2)
            self.assertTrue((tensor_cache_path / "manifest.json").exists())

    def test_tensor_cache_records_untraceable_games_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            tensor_cache_path = root / "cache" / "shogi-policy-value-tensors"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "8d8e"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "illegal move"):
                build_shogi_policy_value_tensor_cache(
                    data_selection_path=data_selection_path,
                    output_path=tensor_cache_path,
                    shard_games=1,
                )

    def test_initializes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            init_checkpoint_path = root / "init.pt"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--checkpoint-path",
                    str(init_checkpoint_path),
                    "--metrics-path",
                    str(root / "init-metrics.json"),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--init-checkpoint-path",
                    str(init_checkpoint_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--learning-rate",
                    "0",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["init_checkpoint_path"], str(init_checkpoint_path))
            init_model = load_shogi_policy_value_checkpoint(init_checkpoint_path)
            trained_model = load_shogi_policy_value_checkpoint(checkpoint_path)
            for key, tensor in init_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, trained_model.state_dict()[key]))

    def test_writes_periodic_checkpoints_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-policy-value",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_policy_value",
                    "--data-selection",
                    str(data_selection_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "3",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--metrics-every",
                    "2",
                    "--eval-every",
                    "2",
                    "--keep-last-n-checkpoints",
                    "2",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            self.assertFalse((root / "checkpoint_step_1.pt").exists())
            self.assertTrue((root / "checkpoint_step_2.pt").exists())
            self.assertTrue((root / "checkpoint_step_3.pt").exists())
            step_metrics_path = root / "metrics_step_2.json"
            self.assertTrue(step_metrics_path.exists())
            step_metrics = json.loads(step_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(step_metrics["step"], 2)
            self.assertEqual(step_metrics["max_steps"], 3)
            self.assertIn("loss", step_metrics)
            self.assertIn("data_wait_seconds", step_metrics)
            self.assertIn("forward_backward_seconds", step_metrics)
            self.assertIn("optimizer_seconds", step_metrics)
            self.assertIn("eval_metrics", step_metrics)
            self.assertIn("loss", step_metrics["eval_metrics"])
            self.assertIn("accuracy", step_metrics["eval_metrics"])

    def test_rejects_unsplit_data_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            data_selection_path = root / "data-selection.json"
            write_shogi_game_records_jsonl(games_path, [_record(("7g7f",), None)])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "bad-unsplit",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "split"):
                load_shogi_policy_value_data_selection(data_selection_path)

    def test_dataset_source_max_games_limits_loaded_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            write_shogi_game_records_jsonl(
                train_games_path,
                [
                    _record(("7g7f", "3c3d"), "black"),
                    _record(("2g2f", "8c8d"), "white"),
                ],
            )
            write_shogi_game_records_jsonl(eval_games_path, [_record(("5g5f", "5c5d"), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "max-games",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path), "max_games": 1}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path), "max_games": 1}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            definition = load_shogi_policy_value_data_selection(data_selection_path)
            train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(definition)

        self.assertEqual(definition.train_sources[0].max_games, 1)
        self.assertEqual(len(train_examples), 2)
        self.assertEqual(len(eval_examples), 2)
        self.assertEqual(
            shogi_policy_value_data_selection_to_json(definition)["train_sources"][0]["max_games"],
            1,
        )

    def test_loads_global_target_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_games_path = root / "teacher-games.jsonl"
            self_play_games_path = root / "self-play-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            write_shogi_game_records_jsonl(
                teacher_games_path,
                [_record_with_multipv_info(("7g7f",), "black")],
            )
            write_shogi_game_records_jsonl(self_play_games_path, [_record(("2g2f",), "black")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("5g5f",), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "target-construction",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "decision_usi_multipv",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "decision_usi_score",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [
                            {"kind": "game_records_jsonl", "path": str(teacher_games_path)},
                            {"kind": "game_records_jsonl", "path": str(self_play_games_path)},
                        ],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            definition = load_shogi_policy_value_data_selection(data_selection_path)
            train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(definition)

        self.assertEqual(definition.target_construction.policy, "decision_usi_multipv")
        self.assertEqual(definition.target_construction.value, "decision_usi_score")
        self.assertEqual(train_examples[0].policy_targets, {"7g7f": 1.0})
        self.assertIsNotNone(train_examples[0].value_target)
        self.assertNotEqual(train_examples[0].value_target, 1.0)
        self.assertIsNone(train_examples[1].policy_targets)
        self.assertIsNone(train_examples[1].value_target)
        self.assertIsNone(eval_examples[0].policy_targets)
        self.assertIsNone(eval_examples[0].value_target)
        self.assertEqual(
            shogi_policy_value_data_selection_to_json(definition)["target_construction"]["policy"],
            "decision_usi_multipv",
        )

    def test_loads_engine_analysis_target_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            analysis_path = root / "analysis.jsonl"
            data_selection_path = root / "data-selection.json"
            train_record = _record(("7g7f",), "black")
            eval_record = _record(("2g2f", "8c8d"), "white")
            train_transition = trace_shogi_game_record(train_record).transitions[0]
            eval_transition = trace_shogi_game_record(eval_record).transitions[1]
            write_shogi_game_records_jsonl(train_games_path, [train_record])
            write_shogi_game_records_jsonl(eval_games_path, [eval_record])
            write_shogi_engine_analysis_jsonl(
                analysis_path,
                [
                    ShogiEngineAnalysis(
                        position_sfen=train_transition.position_sfen,
                        legal_moves=train_transition.legal_moves,
                        engine=WHITE_ACTOR,
                        usi_info_lines=(
                            "info multipv 1 score cp 300 pv 7g7f",
                            "info multipv 2 score cp 0 pv 2g2f",
                        ),
                    ),
                    ShogiEngineAnalysis(
                        position_sfen=eval_transition.position_sfen,
                        legal_moves=eval_transition.legal_moves,
                        engine=WHITE_ACTOR,
                        usi_info_lines=("info multipv 1 score cp -300 pv 8c8d",),
                    ),
                ],
            )
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "engine-analysis-targets",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "engine_analysis_multipv",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "engine_analysis_score",
                            "score_cp_scale": 300.0,
                        },
                        "analysis_sources": [{"kind": "shogi_engine_analysis_jsonl", "path": str(analysis_path)}],
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            definition = load_shogi_policy_value_data_selection(data_selection_path)
            train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(definition)

        self.assertEqual(definition.analysis_sources[0].path, analysis_path)
        self.assertGreater(train_examples[0].policy_targets["7g7f"], train_examples[0].policy_targets["2g2f"])
        self.assertAlmostEqual(train_examples[0].value_target or 0.0, 0.761594, places=5)
        self.assertGreater(eval_examples[0].policy_targets["7g7f"], eval_examples[0].policy_targets["2g2f"])
        self.assertAlmostEqual(eval_examples[0].value_target or 0.0, 0.761594, places=5)
        self.assertEqual(eval_examples[1].policy_targets, {"8c8d": 1.0})
        self.assertAlmostEqual(eval_examples[1].value_target or 0.0, -0.761594, places=5)

    def test_rejects_engine_analysis_targets_without_analysis_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f",), "black")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f",), "black")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "missing-analysis-source",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "engine_analysis_multipv",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "engine_analysis_score",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "analysis_sources"):
                load_shogi_policy_value_data_selection(data_selection_path)

    def test_rejects_example_jsonl_dataset_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_selection_path = root / "data-selection.json"
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "bad-source-kind",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "examples_jsonl", "path": "train-examples.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "game_records_jsonl"):
                load_shogi_policy_value_data_selection(data_selection_path)


if __name__ == "__main__":
    unittest.main()
