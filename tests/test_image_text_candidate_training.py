import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.image_classification import FASHION_MNIST_LABELS, ImageChoiceExample
from intrep.image_text_candidate_training import (
    ImageTextCandidateTrainingConfig,
    train_image_text_candidate_model,
)
from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageTextCandidateTrainingTest(unittest.TestCase):
    def test_trains_image_text_candidate_selector(self) -> None:
        with TemporaryDirectory() as directory:
            result = train_image_text_candidate_model(
                train_examples=_write_examples(Path(directory)),
                text_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                config=ImageTextCandidateTrainingConfig(
                    text_context_length=16,
                    image_patch_size=1,
                    max_steps=2,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=17,
                    model_preset="tiny",
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )

        self.assertIsInstance(result.model, SharedMultimodalModel)
        self.assertEqual(result.metrics.train_case_count, 2)
        self.assertEqual(result.metrics.max_steps, 2)
        self.assertGreater(result.metrics.train_initial_loss, 0.0)
        self.assertGreater(result.metrics.train_final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.train_accuracy, 0.0)
        self.assertLessEqual(result.metrics.train_accuracy, 1.0)


class SharedMultimodalCandidatePathTest(unittest.TestCase):
    def test_outputs_fusion_candidate_logits(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=32,
            text_context_length=4,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )

        logits = model.image_text_fusion_candidate_logits(
            torch.zeros((2, 4, 4), dtype=torch.float32),
            torch.ones((3, 2), dtype=torch.long),
            torch.ones((3, 2), dtype=torch.bool),
        )

        self.assertEqual(logits.shape, torch.Size([2, 3]))


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
