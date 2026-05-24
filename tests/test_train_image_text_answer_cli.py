import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.problems.image_text_answer.checkpoint import load_image_text_answer_checkpoint
from intrep.problems.image_text_answer.examples import ImageTextAnswerExample, image_text_answer_example_to_record
from intrep.problems.image_text_choice.checkpoint import save_image_text_choice_checkpoint
from intrep.problems.image_text_choice.training import ImageTextChoiceTrainingConfig, train_image_text_choice_model
from intrep.problems.image_classification.examples import FASHION_MNIST_LABELS
from intrep.problems.image_text_choice.examples import ImageTextChoiceExample
from intrep.domains.language.tokenizer import build_text_tokenizer, save_text_tokenizer
from intrep.train_image_text_answer import main


class TrainImageTextAnswerCLITest(unittest.TestCase):
    def test_trains_from_image_text_answer_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_a = root / "a.pgm"
            image_b = root / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            train_path = root / "answers.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "answer.pt"
            examples = [
                ImageTextAnswerExample(image_path=image_a, prompt="answer: ", answer_text="black"),
                ImageTextAnswerExample(image_path=image_b, prompt="answer: ", answer_text="white"),
            ]
            train_path.write_text(
                "\n".join(json.dumps(image_text_answer_example_to_record(example)) for example in examples) + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(output):
                main(
                    [
                        "--train-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--text-context-length",
                        "16",
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
                        "--embedding-dim",
                        "8",
                        "--num-heads",
                        "2",
                        "--hidden-dim",
                        "16",
                        "--num-layers",
                        "1",
                        "--device",
                        "cpu",
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_text_answer_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["schema_version"], "intrep.image_text_answer_run.v1")
        self.assertEqual(payload["metrics"]["train_case_count"], 2)
        self.assertEqual(checkpoint.config.text_context_length, 16)
        self.assertIn("intrep image text answer", output.getvalue())
        self.assertIn("train_cases=2", output.getvalue())

    def test_initializes_from_compatible_shared_core_checkpoint(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_a = root / "a.pgm"
            image_b = root / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            choice_examples = [
                ImageTextChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=9),
                ImageTextChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=0),
            ]
            choice_result = train_image_text_choice_model(
                train_examples=choice_examples,
                tokenizer_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                prompt="answer: ",
                config=ImageTextChoiceTrainingConfig(
                    text_context_length=32,
                    image_patch_size=1,
                    max_steps=1,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=17,
                    embedding_dim=8,
                    num_heads=2,
                    hidden_dim=16,
                    num_layers=1,
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )
            choice_checkpoint_path = root / "choice.pt"
            save_image_text_choice_checkpoint(choice_checkpoint_path, choice_result)
            train_path = root / "answers.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "answer.pt"
            answer_examples = [
                ImageTextAnswerExample(image_path=image_a, prompt="answer: ", answer_text="Ankle boot"),
                ImageTextAnswerExample(image_path=image_b, prompt="answer: ", answer_text="T-shirt/top"),
            ]
            train_path.write_text(
                "\n".join(json.dumps(image_text_answer_example_to_record(example)) for example in answer_examples)
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(output):
                main(
                    [
                        "--train-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--init-checkpoint-path",
                        str(choice_checkpoint_path),
                        "--freeze-module",
                        "core",
                        "--freeze-module",
                        "image_input_layer",
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
                        "--embedding-dim",
                        "8",
                        "--num-heads",
                        "2",
                        "--hidden-dim",
                        "16",
                        "--num-layers",
                        "1",
                        "--device",
                        "cpu",
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_text_answer_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["init_checkpoint_path"], str(choice_checkpoint_path))
        self.assertEqual(payload["init_checkpoint_schema"], "intrep.image_text_choice_checkpoint.v1")
        self.assertEqual(payload["frozen_modules"], ["core", "image_input_layer"])
        self.assertEqual(tuple(checkpoint.config.frozen_modules), ("core", "image_input_layer"))
        _assert_state_dict_equal(checkpoint.model.core.state_dict(), choice_result.model.core.state_dict())
        _assert_state_dict_equal(
            checkpoint.model.image_input_layer.state_dict(),
            choice_result.model.image_input_layer.state_dict(),
        )
        self.assertEqual(checkpoint.config.text_context_length, 32)
        self.assertIn("train_cases=2", output.getvalue())

    def test_uses_saved_text_tokenizer(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_a = root / "a.pgm"
            image_b = root / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            train_path = root / "answers.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "answer.pt"
            tokenizer_path = root / "tokenizer.json"
            examples = [
                ImageTextAnswerExample(image_path=image_a, prompt="answer: ", answer_text="black"),
                ImageTextAnswerExample(image_path=image_b, prompt="answer: ", answer_text="white"),
            ]
            train_path.write_text(
                "\n".join(json.dumps(image_text_answer_example_to_record(example)) for example in examples) + "\n",
                encoding="utf-8",
            )
            tokenizer = build_text_tokenizer("answer: black white", kind="byte-pair", vocab_size=270)
            save_text_tokenizer(tokenizer_path, tokenizer)

            main(
                [
                    "--train-path",
                    str(train_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--tokenizer-path",
                    str(tokenizer_path),
                    "--text-context-length",
                    "16",
                    "--image-patch-size",
                    "1",
                    "--batch-size",
                    "2",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_text_answer_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["tokenizer_path"], str(tokenizer_path))
        self.assertEqual(checkpoint.tokenizer.vocab_size, tokenizer.vocab_size)


def _assert_state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key, left_value in left.items():
        torch.testing.assert_close(left_value, right[key])


if __name__ == "__main__":
    unittest.main()
