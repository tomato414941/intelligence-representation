import unittest

import torch
from torch import nn

from intrep.training_utils import build_adamw, build_lr_scheduler, clip_gradients


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

    def test_constant_lr_scheduler_keeps_learning_rate(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = build_lr_scheduler(
            optimizer,
            schedule="constant",
            warmup_steps=0,
            max_steps=4,
        )

        optimizer.step()
        scheduler.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_warmup_cosine_lr_scheduler_warms_up_then_decays(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = build_lr_scheduler(
            optimizer,
            schedule="warmup_cosine",
            warmup_steps=2,
            max_steps=6,
        )

        learning_rates = []
        for _ in range(6):
            learning_rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(learning_rates[0], 0.005)
        self.assertAlmostEqual(learning_rates[1], 0.01)
        self.assertGreater(learning_rates[2], learning_rates[4])

    def test_lr_scheduler_rejects_invalid_values(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        with self.assertRaisesRegex(ValueError, "warmup_steps must be non-negative"):
            build_lr_scheduler(optimizer, schedule="constant", warmup_steps=-1, max_steps=4)
        with self.assertRaisesRegex(ValueError, "max_steps must be non-negative"):
            build_lr_scheduler(optimizer, schedule="constant", warmup_steps=0, max_steps=-1)
        with self.assertRaisesRegex(ValueError, "lr_schedule"):
            build_lr_scheduler(optimizer, schedule="linear", warmup_steps=0, max_steps=4)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
