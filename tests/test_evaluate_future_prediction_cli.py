import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep import evaluate_future_prediction
from intrep.future_prediction_ranking import (
    FuturePredictionRankingMetrics,
    FuturePredictionRankingSummary,
)
from intrep.mixed_corpus import MixedDocument


@dataclass(frozen=True)
class FakeTrainingResult:
    final_train_loss: float = 1.25


@dataclass(frozen=True)
class FakeTrainingArtifacts:
    model: object = object()
    tokenizer: object = object()
    result: FakeTrainingResult = FakeTrainingResult()


class EvaluateFuturePredictionCLITest(unittest.TestCase):
    def test_evaluates_signal_train_and_eval_paths(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            _write_events(train_path, "train")
            _write_events(eval_path, "eval")

            with redirect_stdout(output):
                evaluate_future_prediction.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--target-channel",
                        "consequence",
                        "--condition",
                        "same_modality_negative",
                        "--model-preset",
                        "tiny",
                        "--max-steps",
                        "1",
                        "--context-length",
                        "32",
                        "--batch-size",
                        "2",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        stdout = output.getvalue()
        self.assertIn("intrep future-prediction evaluation", stdout)
        self.assertIn("target_channel=consequence", stdout)
        self.assertIn("generalization_eval=true", stdout)
        self.assertEqual(payload["target_channel"], "consequence")
        self.assertTrue(payload["generalization_eval"])
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)
        self.assertIn("delta_top1_accuracy", payload)
        self.assertIn("delta_margin", payload)
        self.assertEqual(payload["explicit_negative_rate"], 0.0)
        self.assertEqual(payload["no_negative_case_count"], 0)

    def test_eval_path_absent_reports_train_split_not_generalization(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            metrics_path = root / "metrics.json"
            _write_events(train_path, "train")

            with redirect_stdout(output):
                evaluate_future_prediction.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--target-channel",
                        "consequence",
                        "--condition",
                        "same_modality_negative",
                        "--model-preset",
                        "tiny",
                        "--max-steps",
                        "1",
                        "--context-length",
                        "32",
                        "--batch-size",
                        "2",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        stdout = output.getvalue()
        self.assertIn("eval_path=train", stdout)
        self.assertIn("eval_split=train", stdout)
        self.assertIn("generalization_eval=false", stdout)
        self.assertEqual(payload["eval_split"], "train")
        self.assertFalse(payload["generalization_eval"])
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)
        self.assertIn("delta_top1_accuracy", payload)
        self.assertIn("delta_margin", payload)

    def test_payload_ref_train_path_is_rejected_before_training(self) -> None:
        stderr = io.StringIO()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                (
                    '{"channel":"observation",'
                    '"payload_ref":{"uri":"dataset://images/frame-1.png","media_type":"image/png"}}\n'
                    '{"channel":"action","payload":"inspect image"}\n'
                    '{"channel":"consequence","payload":"found object"}\n'
                ),
                encoding="utf-8",
            )

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluate_future_prediction.main(
                        [
                            "--train-path",
                            str(path),
                            "--target-channel",
                            "consequence",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("does not support payload_ref", stderr.getvalue())

    def test_generic_cli_does_not_expose_image_label_target(self) -> None:
        stderr = io.StringIO()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            _write_events(path, "train")

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluate_future_prediction.main(
                        [
                            "--train-path",
                            str(path),
                            "--target-channel",
                            "label",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice: 'label'", stderr.getvalue())

    def test_run_config_passes_image_rendering_options_to_training_and_ranking(self) -> None:
        output = io.StringIO()
        summary = FuturePredictionRankingSummary(
            overall=FuturePredictionRankingMetrics(
                top1_accuracy=0.5,
                mean_positive_loss=1.0,
                mean_best_negative_loss=2.0,
                mean_margin=1.0,
            ),
            by_condition={
                "image_to_label": FuturePredictionRankingMetrics(
                    top1_accuracy=0.5,
                    mean_positive_loss=1.0,
                    mean_best_negative_loss=2.0,
                    mean_margin=1.0,
                )
            },
            condition_counts={"image_to_label": 2},
        )
        ranking_calls: list[dict[str, object]] = []
        render_calls: list[dict[str, object]] = []
        training_call_count = 0

        def fake_evaluate_future_prediction_ranking(
            cases,
            model,
            tokenizer,
            *,
            rendering="signal",
            image_patch_size=1,
            image_channel_bins=4,
            max_negatives=None,
        ):
            del model, tokenizer
            ranking_calls.append(
                {
                    "case_count": len(cases),
                    "rendering": rendering,
                    "image_patch_size": image_patch_size,
                    "image_channel_bins": image_channel_bins,
                    "max_negatives": max_negatives,
                }
            )
            return summary

        def fake_train_mixed_gpt_with_artifacts(
            *,
            documents,
            eval_documents=None,
            training_config,
            model_config,
        ):
            nonlocal training_call_count
            del documents, eval_documents, training_config, model_config
            training_call_count += 1
            return FakeTrainingArtifacts()

        def fake_signals_to_mixed_documents(
            events,
            *,
            render_format="signal-tags",
            image_patch_size=1,
            image_channel_bins=4,
        ):
            render_calls.append(
                {
                    "event_count": len(events),
                    "render_format": render_format,
                    "image_patch_size": image_patch_size,
                    "image_channel_bins": image_channel_bins,
                }
            )
            return [MixedDocument(id="fake", modality="signals", content="fake")]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            _write_image_label_events(train_path, root / "train-images", "train")
            _write_image_label_events(eval_path, root / "eval-images", "eval")

            with (
                patch.object(
                    evaluate_future_prediction,
                    "evaluate_future_prediction_ranking",
                    fake_evaluate_future_prediction_ranking,
                ),
                patch.object(
                    evaluate_future_prediction,
                    "train_mixed_gpt_with_artifacts",
                    fake_train_mixed_gpt_with_artifacts,
                ),
                patch.object(
                    evaluate_future_prediction,
                    "signals_to_mixed_documents",
                    fake_signals_to_mixed_documents,
                ),
                redirect_stdout(output),
            ):
                evaluate_future_prediction.run_future_prediction_evaluation(
                    evaluate_future_prediction.FuturePredictionEvaluationConfig(
                        train_path=train_path,
                        eval_path=eval_path,
                        target_channel="label",
                        rendering="image-tokens",
                        image_patch_size=2,
                        image_channel_bins=8,
                        max_negatives=1,
                        max_steps=1,
                    )
                )

        self.assertEqual(len(ranking_calls), 2)
        self.assertEqual(
            ranking_calls,
            [
                {
                    "case_count": 2,
                    "rendering": "image-tokens",
                    "image_patch_size": 2,
                    "image_channel_bins": 8,
                    "max_negatives": 1,
                },
                {
                    "case_count": 2,
                    "rendering": "image-tokens",
                    "image_patch_size": 2,
                    "image_channel_bins": 8,
                    "max_negatives": 1,
                },
            ],
        )
        self.assertEqual(
            render_calls,
            [
                {
                    "event_count": 4,
                    "render_format": "image-tokens",
                    "image_patch_size": 2,
                    "image_channel_bins": 8,
                },
                {
                    "event_count": 4,
                    "render_format": "image-tokens",
                    "image_patch_size": 2,
                    "image_channel_bins": 8,
                },
            ],
        )
        self.assertEqual(training_call_count, 1)
        self.assertIn("rendering=image-tokens", output.getvalue())
        self.assertIn("max_negatives=1", output.getvalue())


def _write_events(path: Path, prefix: str) -> None:
    rows = [
        {
            "id": f"{prefix}_ep1_obs",
            "channel": "observation",
            "episode_id": f"{prefix}_ep1",
            "time_index": 0,
            "payload": "box_a has key ; box_b has coin",
        },
        {
            "id": f"{prefix}_ep1_action",
            "channel": "action",
            "episode_id": f"{prefix}_ep1",
            "time_index": 1,
            "payload": "open box_a",
        },
        {
            "id": f"{prefix}_ep1_cons",
            "channel": "consequence",
            "episode_id": f"{prefix}_ep1",
            "time_index": 2,
            "payload": "see key",
        },
        {
            "id": f"{prefix}_ep2_obs",
            "channel": "observation",
            "episode_id": f"{prefix}_ep2",
            "time_index": 0,
            "payload": "box_a has key ; box_b has coin",
        },
        {
            "id": f"{prefix}_ep2_action",
            "channel": "action",
            "episode_id": f"{prefix}_ep2",
            "time_index": 1,
            "payload": "open box_b",
        },
        {
            "id": f"{prefix}_ep2_cons",
            "channel": "consequence",
            "episode_id": f"{prefix}_ep2",
            "time_index": 2,
            "payload": "see coin",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_image_label_events(path: Path, image_dir: Path, prefix: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / f"{prefix}_a.pgm"
    image_b = image_dir / f"{prefix}_b.pgm"
    image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
    image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([255, 0]))
    rows = [
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_a.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "9:Ankle boot"},
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_b.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "0:T-shirt/top"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
