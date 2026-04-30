import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.image_classification import ImagePatchInputLayer
from intrep.image_conditioned_text_scoring import (
    choose_image_conditioned_text_candidate,
    score_image_conditioned_text_candidates,
)


class ImageConditionedTextScoringTest(unittest.TestCase):
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

        losses = score_image_conditioned_text_candidates(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4), dtype=torch.float32),
            prompt="?",
            candidates=("a", "b"),
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(loss > 0.0 for loss in losses))

    def test_scores_candidates_without_prompt(self) -> None:
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

        losses = score_image_conditioned_text_candidates(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4), dtype=torch.float32),
            prompt="",
            candidates=("a", "b"),
        )

        self.assertEqual(len(losses), 2)

    def test_scores_rgb_image_candidates(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            channel_count=3,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        losses = score_image_conditioned_text_candidates(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4, 3), dtype=torch.float32),
            prompt="",
            candidates=("a", "b"),
        )

        self.assertEqual(len(losses), 2)

    def test_choose_image_conditioned_text_candidate_returns_lowest_loss_index(self) -> None:
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

        choice = choose_image_conditioned_text_candidate(
            image_input_layer=image_input,
            text_model=text_model,
            tokenizer=tokenizer,
            image=torch.zeros((4, 4), dtype=torch.float32),
            prompt="?",
            candidates=("a", "b"),
        )

        self.assertEqual(choice, 1)

    def test_score_image_conditioned_text_candidates_validates_inputs(self) -> None:
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
            score_image_conditioned_text_candidates(
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                image=torch.zeros((4, 4), dtype=torch.float32),
                prompt="?",
                candidates=(),
            )
        with self.assertRaisesRegex(ValueError, "image"):
            score_image_conditioned_text_candidates(
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                image=torch.zeros((2, 4, 4), dtype=torch.float32),
                prompt="?",
                candidates=("a",),
            )


if __name__ == "__main__":
    unittest.main()
