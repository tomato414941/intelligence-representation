import json
import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.problems.shogi_policy_value.benchmarking import (
    benchmark_shogi_policy_value_inference_batching,
    benchmark_shogi_position_feature_generation,
    latency_summary_ms,
    load_position_sfens_from_jsonl,
    parse_batch_sizes,
)
from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_checkpoint
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, train_shogi_policy_value_model
from intrep.representation.inputs.shogi_position_features.position_rich import SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH, SHOGI_RICH_POSITION_INPUT_SCHEMA_ID
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves


class ShogiPolicyValueBenchmarkingTest(unittest.TestCase):
    def test_latency_summary_reports_millisecond_percentiles(self) -> None:
        summary = latency_summary_ms([0.001, 0.003, 0.002])

        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertAlmostEqual(summary["mean"], 2.0)

    def test_load_position_sfens_from_jsonl_reads_field_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"position_sfen": shogi.Board().sfen()}),
                        json.dumps({"position_sfen": shogi.Board().sfen()}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            position_sfens = load_position_sfens_from_jsonl(path, limit=1)

        self.assertEqual(position_sfens, [shogi.Board().sfen()])

    def test_benchmark_shogi_position_feature_generation_returns_schema_and_timings(self) -> None:
        result = benchmark_shogi_position_feature_generation([shogi.Board().sfen()], warmup=0, repeat=1)

        self.assertEqual(
            result["schema_version"],
            "intrep.problems.shogi_policy_value.position_feature_generation_benchmark.v1",
        )
        self.assertEqual(result["input_schema_id"], SHOGI_RICH_POSITION_INPUT_SCHEMA_ID)
        self.assertEqual(result["input_feature_manifest_hash"], SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH)
        self.assertEqual(result["position_count"], 1)
        self.assertEqual(result["measured_position_count"], 1)
        self.assertGreater(result["latency_ms"]["min"], 0.0)
        self.assertGreater(result["positions_per_second"], 0.0)

    def test_benchmark_shogi_policy_value_inference_batching_uses_checkpoint_batches(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "shogi.pt"
            save_shogi_policy_value_checkpoint(checkpoint_path, result)

            benchmark = benchmark_shogi_policy_value_inference_batching(
                checkpoint_path,
                [example.position_sfen for example in examples],
                batch_sizes=(1, 2),
                warmup_batches=0,
                measure_batches=1,
            )

        self.assertEqual(
            benchmark["schema_version"],
            "intrep.problems.shogi_policy_value.inference_batching_benchmark.v1",
        )
        self.assertEqual(benchmark["input_schema_id"], SHOGI_RICH_POSITION_INPUT_SCHEMA_ID)
        self.assertEqual(benchmark["input_feature_manifest_hash"], SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH)
        self.assertTrue(benchmark["includes_feature_generation"])
        self.assertEqual([entry["batch_size"] for entry in benchmark["batch_results"]], [1, 2])
        self.assertGreater(benchmark["batch_results"][0]["latency_ms"]["min"], 0.0)
        self.assertGreater(benchmark["batch_results"][1]["positions_per_second"], 0.0)

    def test_parse_batch_sizes_rejects_empty_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            parse_batch_sizes("")


if __name__ == "__main__":
    unittest.main()
