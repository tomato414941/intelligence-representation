import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.image_classification import FASHION_MNIST_LABELS, ImageChoiceExample
from intrep.image_text_candidate_training import (
    ImageTextCandidateTrainingConfig,
    evaluate_image_text_candidate_model,
    train_image_text_candidate_model,
)
from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageTextCandidateTrainingTest(unittest.TestCase):
    def test_trains_image_text_candidate_selector(self) -> None:
        with TemporaryDirectory() as directory:
            result = train_image_text_candidate_model(
                train_examples=_write_examples(Path(directory)),
                text_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                prompt="What is this item?",
                config=ImageTextCandidateTrainingConfig(
                    text_context_length=32,
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

            eval_metrics = evaluate_image_text_candidate_model(
                model=result.model,
                tokenizer=result.tokenizer,
                examples=_write_examples(Path(directory)),
                prompt="What is this item?",
            )

        self.assertIsInstance(result.model, SharedMultimodalModel)
        self.assertEqual(result.metrics.train_case_count, 2)
        self.assertEqual(result.metrics.max_steps, 2)
        self.assertGreater(result.metrics.train_initial_loss, 0.0)
        self.assertGreater(result.metrics.train_final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.train_accuracy, 0.0)
        self.assertLessEqual(result.metrics.train_accuracy, 1.0)
        self.assertIsNone(result.metrics.text_initial_loss)
        self.assertIsNone(result.metrics.text_final_loss)
        self.assertEqual(eval_metrics.case_count, 2)
        self.assertGreater(eval_metrics.loss, 0.0)
        self.assertGreaterEqual(eval_metrics.accuracy, 0.0)
        self.assertLessEqual(eval_metrics.accuracy, 1.0)

    def test_can_mix_text_lm_and_image_text_candidate_training(self) -> None:
        with TemporaryDirectory() as directory:
            result = train_image_text_candidate_model(
                train_examples=_write_examples(Path(directory)),
                text_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                language_modeling_corpus="alpha beta gamma alpha beta gamma " * 20,
                prompt="What is this item?",
                config=ImageTextCandidateTrainingConfig(
                    text_context_length=32,
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

        self.assertGreater(result.metrics.train_final_loss, 0.0)
        self.assertIsNotNone(result.metrics.text_initial_loss)
        self.assertIsNotNone(result.metrics.text_final_loss)
        assert result.metrics.text_initial_loss is not None
        assert result.metrics.text_final_loss is not None
        self.assertGreater(result.metrics.text_initial_loss, 0.0)
        self.assertGreater(result.metrics.text_final_loss, 0.0)


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
        )

        logits = model.image_text_fusion_candidate_logits(
            torch.zeros((2, 4, 4), dtype=torch.float32),
            torch.ones((1,), dtype=torch.long),
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
