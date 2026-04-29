import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep import train_signal_text
from intrep.signals import PayloadRef


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_loss: float = 2.0
    final_loss: float = 1.0
    steps: int = 3
    token_count: int = 12
    loss_history: tuple[float, ...] = (2.0, 1.5, 1.0)
    initial_train_loss: float = 2.0
    final_train_loss: float = 1.0
    initial_eval_loss: float | None = None
    final_eval_loss: float | None = None
    eval_split: str = "train"
    generalization_eval: bool = False
    warnings: tuple[str, ...] = ()
    device: str = "cpu"

    @property
    def best_loss(self) -> float:
        return min(self.loss_history)

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def loss_reduction_ratio(self) -> float:
        return self.loss_reduction / self.initial_loss


@dataclass(frozen=True)
class FakeArtifacts:
    result: FakeTrainingResult = FakeTrainingResult()
    model: object = object()
    tokenizer: object = object()


class TrainSignalTextCLITest(unittest.TestCase):
    def test_trains_rendered_text_signals_directly(self) -> None:
        captured: dict[str, object] = {}

        def fake_train_rendered_gpt_with_artifacts(
            *,
            corpus,
            eval_corpus=None,
            training_config,
            model_config,
        ):
            captured["corpus"] = corpus
            captured["eval_corpus"] = eval_corpus
            captured["training_config"] = training_config
            captured["model_config"] = model_config
            return FakeArtifacts()

        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            summary_path = root / "summary.json"
            train_path.write_text(
                '{"channel":"observation","payload":"A.."}\n',
                encoding="utf-8",
            )
            eval_path.write_text(
                '{"channel":"observation","payload":"B.."}\n',
                encoding="utf-8",
            )

            with patch.object(
                train_signal_text,
                "train_rendered_gpt_with_artifacts",
                fake_train_rendered_gpt_with_artifacts,
            ):
                with redirect_stdout(output):
                    train_signal_text.main(
                        [
                            "--train-path",
                            str(train_path),
                            "--eval-path",
                            str(eval_path),
                            "--max-steps",
                            "3",
                            "--model-preset",
                            "tiny",
                            "--loss-summary",
                            "--run-summary-path",
                            str(summary_path),
                        ]
                    )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertIn('<SIGNAL channel="observation">', str(captured["corpus"]))
        self.assertIn('<SIGNAL channel="observation">', str(captured["eval_corpus"]))
        self.assertEqual(captured["training_config"].max_steps, 3)
        self.assertEqual(captured["model_config"].embedding_dim, 8)
        self.assertIn("intrep signal-text training", output.getvalue())
        self.assertIn("loss initial=2.0000 final=1.0000", output.getvalue())
        self.assertEqual(payload["kind"], "train_signal_text")

    def test_rejects_payload_refs(self) -> None:
        stderr = io.StringIO()
        with TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            ref = PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png")
            train_path.write_text(
                (
                    '{"channel":"image","payload_ref":{'
                    f'"uri":"{ref.uri}","media_type":"{ref.media_type}"'
                    '}}\n'
                ),
                encoding="utf-8",
            )

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    train_signal_text.main(["--train-path", str(train_path)])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("signal text training does not support payload_ref", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
