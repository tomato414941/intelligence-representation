from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from intrep import train_gpt
from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_loss: float = 4.0
    final_loss: float = 2.5
    steps: int = 3
    token_count: int = 123
    best_loss: float = 2.25
    loss_reduction: float = 1.5
    loss_reduction_ratio: float = 0.375
    loss_history: tuple[float, ...] = (4.0, 3.0, 2.5)
    initial_train_loss: float | None = 4.25
    final_train_loss: float | None = 2.75
    initial_eval_loss: float | None = 4.5
    final_eval_loss: float | None = 3.5
    device: str = "cpu"


class TrainGPTCLITest(unittest.TestCase):
    def test_default_uses_builtin_corpus(self) -> None:
        captured_documents = object()

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 3)
            self.assertIsNone(training_config.batch_stride)
            self.assertIsNotNone(model_config)
            self.assertEqual(model_config.embedding_dim, 32)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--max-steps", "3"])

        self.assertIsNone(captured_documents)
        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())
        self.assertIn("train_avg_initial=4.2500 train_avg_final=2.7500", output.getvalue())

    def test_model_preset_tiny_passes_model_config(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            self.assertIsNotNone(model_config)
            self.assertEqual(model_config.embedding_dim, 8)
            self.assertEqual(model_config.num_heads, 2)
            self.assertEqual(model_config.hidden_dim, 16)
            return FakeTrainingResult()

        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(io.StringIO()):
                train_gpt.main(["--model-preset", "tiny"])

    def test_model_overrides_pass_model_config(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            self.assertIsNotNone(model_config)
            self.assertEqual(model_config.embedding_dim, 24)
            self.assertEqual(model_config.num_heads, 3)
            self.assertEqual(model_config.hidden_dim, 48)
            self.assertEqual(model_config.num_layers, 2)
            self.assertEqual(model_config.dropout, 0.1)
            return FakeTrainingResult()

        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(io.StringIO()):
                train_gpt.main(
                    [
                        "--model-preset",
                        "tiny",
                        "--embedding-dim",
                        "24",
                        "--num-heads",
                        "3",
                        "--hidden-dim",
                        "48",
                        "--num-layers",
                        "2",
                        "--dropout",
                        "0.1",
                    ]
                )

    def test_invalid_model_config_reports_cli_error(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit):
                train_gpt.main(["--embedding-dim", "10", "--num-heads", "3"])

        self.assertIn("embedding_dim must be divisible by num_heads", error_output.getvalue())

    def test_batch_stride_passes_to_training_config(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            self.assertEqual(training_config.batch_stride, 5)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--batch-stride", "5"])

        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())

    def test_device_and_checkpoint_path_pass_to_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "gpt.pt"

            def fake_train_mixed_gpt(
                *, documents=None, eval_documents=None, training_config, model_config=None
            ):
                self.assertEqual(training_config.device, "auto")
                self.assertEqual(training_config.checkpoint_path, checkpoint_path)
                return FakeTrainingResult(device="cuda")

            output = io.StringIO()
            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(
                        [
                            "--device",
                            "auto",
                            "--checkpoint-path",
                            str(checkpoint_path),
                        ]
                    )

        self.assertIn("device=cuda", output.getvalue())

    def test_unavailable_cuda_reports_cli_error(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            raise ValueError("CUDA device requested but torch.cuda.is_available() is false")

        error_output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stderr(error_output):
                with self.assertRaises(SystemExit):
                    train_gpt.main(["--device", "cuda"])

        self.assertIn("CUDA device requested", error_output.getvalue())

    def test_builtin_grid_corpus_uses_grid_action_documents(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--corpus", "builtin-grid"])

        self.assertIsNotNone(captured_documents)
        assert captured_documents is not None
        modalities = {document.modality for document in captured_documents}
        self.assertIn("grid", modalities)
        self.assertIn("action_log", modalities)
        self.assertIn("next_grid", modalities)
        self.assertIn("next_text", modalities)
        self.assertTrue(any("<action>" in document.content for document in captured_documents))
        self.assertIn("corpus=builtin-grid eval_corpus=none tokens=123 steps=3", output.getvalue())

    def test_file_corpus_uses_document_loader(self) -> None:
        loaded_documents = [MixedDocument(id="custom", modality="text", content="hello")]
        captured_documents: list[MixedDocument] | None = None
        captured_path: Path | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            nonlocal captured_path
            captured_path = Path(path)
            return loaded_documents

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(
                    ["--corpus", "file", "--corpus-path", "corpus.jsonl"],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_path, Path("corpus.jsonl"))
        self.assertEqual(captured_documents, loaded_documents)
        self.assertIn("corpus=corpus.jsonl", output.getvalue())

    def test_file_corpus_requires_path(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                train_gpt.main(["--corpus", "file"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--corpus-path is required", error_output.getvalue())

    def test_eval_corpus_uses_document_loader(self) -> None:
        train_documents = [MixedDocument(id="train", modality="text", content="train")]
        eval_documents = [MixedDocument(id="eval", modality="text", content="eval")]
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            if Path(path) == Path("train.jsonl"):
                return train_documents
            if Path(path) == Path("eval.jsonl"):
                return eval_documents
            raise AssertionError(path)

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_eval_documents
            self.assertEqual(documents, train_documents)
            captured_eval_documents = eval_documents
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(
                    [
                        "--corpus",
                        "file",
                        "--corpus-path",
                        "train.jsonl",
                        "--eval-corpus-path",
                        "eval.jsonl",
                    ],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_eval_documents, eval_documents)
        self.assertIn("eval_corpus=eval.jsonl", output.getvalue())

    def test_loss_summary_prints_compact_line(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--loss-summary"])

        self.assertIn(
            "loss initial=4.0000 final=2.5000 best=2.2500 delta=1.5000 ratio=37.50%"
            " eval_initial=4.5000 eval_final=3.5000"
            " train_avg_initial=4.2500 train_avg_final=2.7500",
            output.getvalue(),
        )

    def test_loss_history_path_writes_json(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            return FakeTrainingResult()

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            loss_history_path = Path(temp_dir) / "loss-history.json"
            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(["--loss-history-path", str(loss_history_path)])

            payload = json.loads(loss_history_path.read_text(encoding="utf-8"))

        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())
        self.assertEqual(payload["steps"], 3)
        self.assertEqual(payload["token_count"], 123)
        self.assertIsNone(payload["batch_stride"])
        self.assertEqual(payload["initial_loss"], 4.0)
        self.assertEqual(payload["final_loss"], 2.5)
        self.assertEqual(payload["initial_step_loss"], 4.0)
        self.assertEqual(payload["final_step_loss"], 2.5)
        self.assertEqual(payload["best_loss"], 2.25)
        self.assertEqual(payload["best_step_loss"], 2.25)
        self.assertEqual(payload["loss_history"], [4.0, 3.0, 2.5])
        self.assertEqual(payload["initial_train_loss"], 4.25)
        self.assertEqual(payload["final_train_loss"], 2.75)
        self.assertEqual(payload["initial_eval_loss"], 4.5)
        self.assertEqual(payload["final_eval_loss"], 3.5)
        self.assertIsNone(payload["eval_split"])
        self.assertIsNone(payload["generalization_eval"])

    def test_run_summary_path_writes_normalized_summary(self) -> None:
        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            return FakeTrainingResult()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_summary_path = Path(temp_dir) / "run-summary.json"
            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(io.StringIO()):
                    train_gpt.main(
                        [
                            "--run-id",
                            "run-1",
                            "--run-summary-path",
                            str(run_summary_path),
                        ]
                    )

            payload = json.loads(run_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "intrep.run_summary.v1")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["kind"], "train_gpt")
        self.assertEqual(payload["config"]["model"]["embedding_dim"], 32)
        self.assertAlmostEqual(
            payload["metrics"]["language_modeling"]["final_train_perplexity"],
            15.642631884188171,
        )

    def test_file_corpus_eval_corpus_and_loss_history_use_real_jsonl_loader(self) -> None:
        train_documents = [
            MixedDocument(id="train_text_001", modality="text", content="train observation"),
            MixedDocument(
                id="train_env_001",
                modality="environment_symbolic",
                content="<obs> box closed <action> open box <next_obs> key visible",
            ),
        ]
        eval_documents = [
            MixedDocument(id="eval_text_001", modality="text", content="eval observation"),
        ]
        captured_documents: list[MixedDocument] | None = None
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_documents, captured_eval_documents
            captured_documents = documents
            captured_eval_documents = eval_documents
            self.assertEqual(training_config.max_steps, 2)
            return FakeTrainingResult(
                steps=2,
                token_count=456,
                initial_eval_loss=5.5,
                final_eval_loss=4.25,
            )

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / "train.jsonl"
            eval_corpus_path = temp_path / "eval.jsonl"
            loss_history_path = temp_path / "loss-history.json"
            write_mixed_documents_jsonl(corpus_path, train_documents)
            write_mixed_documents_jsonl(eval_corpus_path, eval_documents)

            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(
                        [
                            "--corpus",
                            "file",
                            "--corpus-path",
                            str(corpus_path),
                            "--eval-corpus-path",
                            str(eval_corpus_path),
                            "--max-steps",
                            "2",
                            "--batch-stride",
                            "7",
                            "--loss-summary",
                            "--loss-history-path",
                            str(loss_history_path),
                        ]
                    )

            payload = json.loads(loss_history_path.read_text(encoding="utf-8"))

        self.assertEqual(captured_documents, train_documents)
        self.assertEqual(captured_eval_documents, eval_documents)
        stdout = output.getvalue()
        self.assertIn(f"corpus={corpus_path}", stdout)
        self.assertIn(f"eval_corpus={eval_corpus_path}", stdout)
        self.assertIn("tokens=456 steps=2 initial_loss=4.0000 final_loss=2.5000", stdout)
        self.assertIn("eval_split=held_out generalization_eval=true", stdout)
        self.assertIn("train_avg_initial=4.2500 train_avg_final=2.7500", stdout)
        self.assertIn("eval_initial=5.5000 eval_final=4.2500", stdout)
        self.assertEqual(payload["batch_stride"], 7)
        self.assertEqual(payload["initial_eval_loss"], 5.5)
        self.assertEqual(payload["final_eval_loss"], 4.25)

    def test_file_corpus_can_load_signal_jsonl_with_signal_rendering(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_train_mixed_gpt(
            *, documents=None, eval_documents=None, training_config, model_config=None
        ):
            nonlocal captured_documents
            captured_documents = documents
            return FakeTrainingResult()

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "signal.jsonl"
            corpus_path.write_text(
                (
                    '{"id":"ep1_obs","role":"observation","modality":"grid",'
                    '"episode_id":"ep1","time_index":0,"content":"A.."}\n'
                    '{"id":"ep1_action","role":"action","modality":"grid_action",'
                    '"episode_id":"ep1","time_index":1,"content":"right"}\n'
                ),
                encoding="utf-8",
            )

            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(
                        [
                            "--corpus",
                            "file",
                            "--corpus-path",
                            str(corpus_path),
                            "--corpus-format",
                            "signal",
                            "--render-format",
                            "signal-tags",
                        ]
                    )

        assert captured_documents is not None
        self.assertEqual(captured_documents[0].modality, "observation")
        self.assertIn('channel="observation"', captured_documents[0].content)
        self.assertIn("corpus_format=signal render_format=signal-tags", output.getvalue())

    def test_file_corpus_rejects_payload_ref_signal_jsonl(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "signal.jsonl"
            corpus_path.write_text(
                (
                    '{"channel":"image",'
                    '"payload_ref":{"uri":"dataset://images/frame-1.png","media_type":"image/png"}}\n'
                ),
                encoding="utf-8",
            )

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    train_gpt.main(
                        [
                            "--corpus",
                            "file",
                            "--corpus-path",
                            str(corpus_path),
                            "--corpus-format",
                            "signal",
                            "--render-format",
                            "signal-tags",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires a channel-specific loader or encoder", stderr.getvalue())

if __name__ == "__main__":
    unittest.main()
