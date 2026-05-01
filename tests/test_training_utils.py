import unittest

import torch
from torch import nn

from intrep.training_utils import build_adamw, clip_gradients


class TrainingUtilsTest(unittest.TestCase):
    def test_build_adamw_excludes_bias_norm_and_embedding_from_weight_decay(self) -> None:
        model = nn.Sequential(
            nn.Embedding(8, 4),
            nn.LayerNorm(4),
            nn.Linear(4, 2),
        )

        optimizer = build_adamw(model, learning_rate=0.001, weight_decay=0.01)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertEqual(len(optimizer.param_groups[1]["params"]), 4)

    def test_clip_gradients_returns_none_when_disabled(self) -> None:
        model = nn.Linear(2, 1)

        grad_norm = clip_gradients(model, None)

        self.assertIsNone(grad_norm)

    def test_clip_gradients_rejects_non_positive_norm(self) -> None:
        model = nn.Linear(2, 1)

        with self.assertRaisesRegex(ValueError, "max_norm must be positive"):
            clip_gradients(model, 0.0)

    def test_clip_gradients_clips_parameter_gradients(self) -> None:
        model = nn.Linear(2, 1)
        loss = model(torch.ones((4, 2))).sum()
        loss.backward()

        grad_norm = clip_gradients(model, 0.1)

        self.assertIsNotNone(grad_norm)
        clipped_norm = torch.linalg.vector_norm(
            torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()])
        )
        self.assertLessEqual(float(clipped_norm), 0.1001)


if __name__ == "__main__":
    unittest.main()
