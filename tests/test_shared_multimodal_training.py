import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_classification import FASHION_MNIST_LABELS, ImageChoiceExample
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.shared_multimodal_training import (
    SharedMultimodalTrainingConfig,
    train_shared_multimodal_model,
)


class SharedMultimodalTrainingTest(unittest.TestCase):
    def test_trains_text_lm_and_image_classification_with_one_model(self) -> None:
        with TemporaryDirectory() as directory:
            result = train_shared_multimodal_model(
                text_corpus="alpha beta gamma alpha beta gamma " * 20,
                image_train_examples=_write_examples(Path(directory)),
                config=SharedMultimodalTrainingConfig(
                    text_context_length=8,
                    image_patch_size=1,
                    max_steps=4,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=17,
                    model_preset="tiny",
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )

        self.assertIsInstance(result.model, SharedMultimodalModel)
        self.assertEqual(result.metrics.max_steps, 4)
        self.assertEqual(result.metrics.image_train_case_count, 2)
        self.assertGreater(result.metrics.text_token_count, 8)
        self.assertGreater(result.metrics.text_initial_loss, 0.0)
        self.assertGreater(result.metrics.text_final_loss, 0.0)
        self.assertGreater(result.metrics.image_initial_loss, 0.0)
        self.assertGreater(result.metrics.image_final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.image_train_accuracy, 0.0)
        self.assertLessEqual(result.metrics.image_train_accuracy, 1.0)

    def test_rejects_empty_text_corpus(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "text_corpus must not be empty"):
                train_shared_multimodal_model(
                    text_corpus="",
                    image_train_examples=_write_examples(Path(directory)),
                    config=SharedMultimodalTrainingConfig(device="cpu"),
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
