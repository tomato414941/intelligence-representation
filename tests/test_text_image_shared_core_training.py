import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_classification import FASHION_MNIST_LABELS, ImageChoiceExample
from intrep.text_image_shared_core_training import (
    TextImageSharedCoreTrainingConfig,
    TextImageSharedCoreModel,
    train_text_image_shared_core_with_result,
)


class TextImageSharedCoreTrainingTest(unittest.TestCase):
    def test_trains_text_and_image_tasks_through_one_shared_core(self) -> None:
        with TemporaryDirectory() as directory:
            examples = _write_examples(Path(directory))

            result = train_text_image_shared_core_with_result(
                text_corpus="alpha beta gamma alpha beta gamma " * 20,
                image_train_examples=examples,
                image_eval_examples=examples,
                config=TextImageSharedCoreTrainingConfig(
                    text_context_length=8,
                    patch_size=1,
                    max_steps=4,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=13,
                    model_preset="tiny",
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )

        self.assertIsInstance(result.model, TextImageSharedCoreModel)
        self.assertTrue(result.metrics.shared_core)
        self.assertEqual(result.metrics.image_train_case_count, 2)
        self.assertEqual(result.metrics.image_eval_case_count, 2)
        self.assertGreater(result.metrics.text_token_count, 8)
        self.assertGreater(result.metrics.text_initial_loss, 0.0)
        self.assertGreater(result.metrics.text_final_loss, 0.0)
        self.assertGreater(result.metrics.image_initial_loss, 0.0)
        self.assertGreater(result.metrics.image_final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.image_train_accuracy, 0.0)
        self.assertLessEqual(result.metrics.image_train_accuracy, 1.0)

    def test_rejects_too_short_text_corpus(self) -> None:
        with TemporaryDirectory() as directory:
            examples = _write_examples(Path(directory))

            with self.assertRaisesRegex(ValueError, "token_ids must be longer than context_length"):
                train_text_image_shared_core_with_result(
                    text_corpus="short",
                    image_train_examples=examples,
                    config=TextImageSharedCoreTrainingConfig(
                        text_context_length=128,
                        patch_size=1,
                        max_steps=1,
                        batch_size=2,
                        device="cpu",
                    ),
                )


def _write_examples(root: Path) -> list[ImageChoiceExample]:
    image_a = root / "a.pgm"
    image_b = root / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    return [
        ImageChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=9),
        ImageChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=0),
    ]


if __name__ == "__main__":
    unittest.main()
