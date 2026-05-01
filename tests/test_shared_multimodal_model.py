import unittest
from unittest.mock import patch

import torch

from intrep.shared_multimodal_model import SharedMultimodalModel


class SharedMultimodalModelTest(unittest.TestCase):
    def test_outputs_text_and_choice_logits_through_one_core(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=32,
            text_context_length=8,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )

        text_logits = model.text_logits(torch.zeros((2, 8), dtype=torch.long))
        choice_logits = model.image_text_choice_logits(
            torch.zeros((2, 4, 4), dtype=torch.float32),
            torch.ones((1,), dtype=torch.long),
            torch.ones((3, 2), dtype=torch.long),
            torch.ones((3, 2), dtype=torch.bool),
        )

        self.assertEqual(text_logits.shape, torch.Size([2, 8, 32]))
        self.assertEqual(choice_logits.shape, torch.Size([2, 3]))

    def test_text_and_choice_paths_use_the_same_core_with_task_masking(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=32,
            text_context_length=8,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )

        with patch.object(model.core, "forward", wraps=model.core.forward) as forward:
            model.text_logits(torch.zeros((2, 8), dtype=torch.long))
            model.image_text_choice_logits(
                torch.zeros((2, 4, 4), dtype=torch.float32),
                torch.ones((2,), dtype=torch.long),
                torch.ones((3, 2), dtype=torch.long),
                torch.ones((3, 2), dtype=torch.bool),
            )

        self.assertEqual(forward.call_count, 2)
        self.assertIs(forward.call_args_list[0].kwargs["causal"], True)
        self.assertIs(forward.call_args_list[1].kwargs["causal"], False)
        self.assertEqual(forward.call_args_list[1].args[0].shape, torch.Size([6, 8, 8]))


if __name__ == "__main__":
    unittest.main()
