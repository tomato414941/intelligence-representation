import unittest

import torch

from intrep.shared_multimodal_model import SharedMultimodalModel


class SharedMultimodalModelTest(unittest.TestCase):
    def test_outputs_text_and_image_logits_through_one_core(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=32,
            text_context_length=8,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )

        text_logits = model.text_logits(torch.zeros((2, 8), dtype=torch.long))
        image_logits = model.image_logits(torch.zeros((2, 4, 4), dtype=torch.float32))

        self.assertEqual(text_logits.shape, torch.Size([2, 8, 32]))
        self.assertEqual(image_logits.shape, torch.Size([2, 10]))


if __name__ == "__main__":
    unittest.main()
