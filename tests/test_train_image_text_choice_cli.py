import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_classification import FASHION_MNIST_LABELS, ImageTextChoiceExample, image_text_choice_example_to_record
from intrep.image_text_answer_checkpoint import save_image_text_answer_checkpoint
from intrep.image_text_answer_training import ImageTextAnswerExample, ImageTextAnswerTrainingConfig, train_image_text_answer_model
from intrep.image_text_choice_checkpoint import load_image_text_choice_checkpoint
from intrep.train_image_text_choice import main


class TrainImageTextChoiceCLITest(unittest.TestCase):
    def test_trains_from_image_text_choice_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "choices.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "choice.pt"
            _write_examples(train_path, root)

            with redirect_stdout(output):
                main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--prompt",
                        "What is this item?",
                        "--text-context-length",
                        "32",
                        "--image-patch-size",
                        "1",
                        "--batch-size",
                        "2",
                        "--max-steps",
                        "1",
                        "--learning-rate",
                        "0.01",
                        "--seed",
                        "17",
                        "--model-preset",
                        "tiny",
                        "--device",
                        "cpu",
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_text_choice_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["schema_version"], "intrep.image_text_choice_run.v1")
        self.assertEqual(payload["metrics"]["train_case_count"], 2)
        self.assertEqual(payload["metrics"]["eval_case_count"], 2)
        self.assertEqual(checkpoint.config.text_context_length, 32)
        self.assertIn("intrep image text choice", output.getvalue())
        self.assertIn("train_cases=2", output.getvalue())

    def test_initializes_from_image_text_answer_checkpoint(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_a = root / "a.pgm"
            image_b = root / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            answer_examples = [
                ImageTextAnswerExample(image_path=image_a, prompt="answer: ", answer_text="Ankle boot"),
                ImageTextAnswerExample(image_path=image_b, prompt="answer: ", answer_text="T-shirt/top"),
            ]
            answer_result = train_image_text_answer_model(
                train_examples=answer_examples,
                tokenizer_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                config=ImageTextAnswerTrainingConfig(
                    text_context_length=32,
                    image_patch_size=1,
                    max_steps=1,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=17,
                    model_preset="tiny",
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )
            answer_checkpoint_path = root / "answer.pt"
            save_image_text_answer_checkpoint(answer_checkpoint_path, answer_result)
            train_path = root / "choices.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "choice.pt"
            _write_examples(train_path, root)

            with redirect_stdout(output):
                main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--init-checkpoint-path",
                        str(answer_checkpoint_path),
                        "--prompt",
                        "answer: ",
                        "--text-context-length",
                        "32",
                        "--image-patch-size",
                        "1",
                        "--batch-size",
                        "2",
                        "--max-steps",
                        "1",
                        "--learning-rate",
                        "0.01",
                        "--seed",
                        "17",
                        "--model-preset",
                        "tiny",
                        "--device",
                        "cpu",
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_text_choice_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["init_checkpoint_path"], str(answer_checkpoint_path))
        self.assertEqual(payload["init_checkpoint_schema"], "intrep.image_text_answer_checkpoint.v1")
        self.assertEqual(checkpoint.config.text_context_length, 32)
        self.assertIn("train_cases=2", output.getvalue())


def _write_examples(path: Path, root: Path) -> None:
    image_a = root / "a.pgm"
    image_b = root / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    examples = [
        ImageTextChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=9),
        ImageTextChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=0),
    ]
    path.write_text(
        "\n".join(json.dumps(image_text_choice_example_to_record(example)) for example in examples) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
