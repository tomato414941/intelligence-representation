from __future__ import annotations

import unittest

from intrep.gpt_model import build_gpt_config


class GPTModelConfigTest(unittest.TestCase):
    def test_small_preset_matches_current_default_shape(self) -> None:
        config = build_gpt_config(preset="small", vocab_size=256, context_length=64)

        self.assertEqual(config.embedding_dim, 32)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.dropout, 0.0)

    def test_tiny_preset_uses_lightweight_shape(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=256, context_length=32)

        self.assertEqual(config.context_length, 32)
        self.assertEqual(config.embedding_dim, 8)
        self.assertEqual(config.num_heads, 2)
        self.assertEqual(config.hidden_dim, 16)

    def test_overrides_take_precedence_over_preset(self) -> None:
        config = build_gpt_config(
            preset="tiny",
            vocab_size=256,
            context_length=16,
            embedding_dim=24,
            num_heads=3,
            hidden_dim=48,
            num_layers=2,
            dropout=0.1,
        )

        self.assertEqual(config.embedding_dim, 24)
        self.assertEqual(config.num_heads, 3)
        self.assertEqual(config.hidden_dim, 48)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.1)

    def test_validates_model_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model preset"):
            build_gpt_config(preset="missing", vocab_size=256, context_length=8)
        with self.assertRaisesRegex(ValueError, "embedding_dim must be positive"):
            build_gpt_config(preset="small", vocab_size=256, context_length=8, embedding_dim=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            build_gpt_config(preset="small", vocab_size=256, context_length=8, dropout=1.0)
        with self.assertRaisesRegex(ValueError, "embedding_dim must be divisible by num_heads"):
            build_gpt_config(
                preset="small",
                vocab_size=256,
                context_length=8,
                embedding_dim=10,
                num_heads=3,
            )


if __name__ == "__main__":
    unittest.main()
