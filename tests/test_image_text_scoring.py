import unittest
import tempfile
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer
from intrep.image_text_scoring import (
    choose_image_text_candidate,
    evaluate_image_text_choices,
    score_image_text_candidates,
)


class ImageTextScoringTest(unittest.TestCase):
    def test_evaluates_image_text_choice_examples(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )
        preferred_id = tokenizer.encode("b")[0]
        with torch.no_grad():
            text_model.token_output.output.weight.zero_()
            text_model.token_output.output.bias.zero_()
            text_model.token_output.output.bias[preferred_id] = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            metrics = evaluate_image_text_choices(
                examples=(
                    ImageChoiceExample(
                        image_path=image_path,
                        choices=("a", "b"),
                        answer_index=1,
                    ),
                ),
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
            )

        self.assertEqual(metrics.case_count, 1)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.predicted_indices, (1,))
        self.assertEqual(len(metrics.losses), 1)
        self.assertEqual(len(metrics.losses[0]), 2)
        self.assertEqual(metrics.to_dict()["predicted_indices"], [1])

    def test_scores_image_conditioned_text_candidates(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        losses = score_image_text_candidates(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4), dtype=torch.float32),
            prompt="?",
            candidates=("a", "b"),
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(loss > 0.0 for loss in losses))

    def test_choose_image_text_candidate_returns_lowest_loss_index(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )
        candidate_ids = tokenizer.encode("b")
        preferred_id = candidate_ids[0]
        with torch.no_grad():
            text_model.token_output.output.weight.zero_()
            text_model.token_output.output.bias.zero_()
            text_model.token_output.output.bias[preferred_id] = 10.0

        choice = choose_image_text_candidate(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4), dtype=torch.float32),
            prompt="?",
            candidates=("a", "b"),
        )

        self.assertEqual(choice, 1)

    def test_score_image_text_candidates_validates_inputs(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        with self.assertRaisesRegex(ValueError, "candidates"):
            score_image_text_candidates(
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                image=torch.zeros((4, 4), dtype=torch.float32),
                prompt="?",
                candidates=(),
            )
        with self.assertRaisesRegex(ValueError, "image"):
            score_image_text_candidates(
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                image=torch.zeros((2, 4, 4), dtype=torch.float32),
                prompt="?",
                candidates=("a",),
            )


if __name__ == "__main__":
    unittest.main()
