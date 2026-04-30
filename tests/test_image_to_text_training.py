import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_classification import ImageChoiceExample
from intrep.image_to_text_training import ImageToTextTrainingConfig, train_image_to_text_labels


class ImageToTextTrainingTest(unittest.TestCase):
    def test_trains_image_to_text_labels(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 0, 0, 0]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 255, 255, 255]))
            examples = [
                ImageChoiceExample(image_path=image_a, choices=("a", "b"), answer_index=0),
                ImageChoiceExample(image_path=image_b, choices=("a", "b"), answer_index=1),
            ]

            metrics = train_image_to_text_labels(
                train_examples=examples,
                eval_examples=examples,
                config=ImageToTextTrainingConfig(
                    patch_size=1,
                    max_steps=20,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=7,
                    model_preset="tiny",
                    device="cpu",
                ),
            )

        self.assertEqual(metrics.target, "answer_text")
        self.assertEqual(metrics.input_representation, "image-patches")
        self.assertEqual(metrics.output_representation, "text-tokens")
        self.assertLess(metrics.train_final_loss, metrics.train_initial_loss)
        self.assertIsNotNone(metrics.eval_final_loss)

    def test_rejects_eval_images_with_different_shape(self) -> None:
        with TemporaryDirectory() as directory:
            train_image = Path(directory) / "train.pgm"
            eval_image = Path(directory) / "eval.pgm"
            train_image.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 0, 0, 0]))
            eval_image.write_bytes(b"P5\n1 2\n255\n" + bytes([0, 0]))
            train_examples = [
                ImageChoiceExample(image_path=train_image, choices=("a", "b"), answer_index=0),
            ]
            eval_examples = [
                ImageChoiceExample(image_path=eval_image, choices=("a", "b"), answer_index=0),
            ]

            with self.assertRaisesRegex(ValueError, "same shape"):
                train_image_to_text_labels(
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    config=ImageToTextTrainingConfig(patch_size=1, max_steps=0, device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
