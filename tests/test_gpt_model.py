from __future__ import annotations

import unittest

import torch

from intrep.gpt_model import CausalTextModel, TokenOutputHead, build_gpt_config
from intrep.transformer_core import SharedTransformerCore


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

    def test_forward_validates_token_ids(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)

        with self.assertRaisesRegex(ValueError, "rank-2"):
            model(torch.tensor([1, 2], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "context_length"):
            model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "vocabulary range"):
            model(torch.tensor([[1, 8]], dtype=torch.long))

    def test_model_exposes_input_embedding_sequence_path(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)
        token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        embeddings = model.embed_tokens(token_ids)
        encoded = model.encode_embeddings(embeddings)

        self.assertEqual(embeddings.shape, torch.Size([1, 4, config.embedding_dim]))
        self.assertEqual(encoded.shape, torch.Size([1, 4, config.embedding_dim]))

    def test_model_uses_shared_transformer_core(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)

        self.assertIsInstance(model.core, SharedTransformerCore)

    def test_model_exposes_token_output_head(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)
        hidden = torch.zeros((1, 4, config.embedding_dim), dtype=torch.float32)

        logits = model.token_logits(hidden)

        self.assertIsInstance(model.token_output, TokenOutputHead)
        self.assertEqual(logits.shape, torch.Size([1, 4, config.vocab_size]))

    def test_encode_embeddings_validates_input_embedding_sequence_shape(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.encode_embeddings(torch.zeros((4, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model.encode_embeddings(torch.zeros((1, 4, config.embedding_dim), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "context_length"):
            model.encode_embeddings(torch.zeros((1, 5, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            model.encode_embeddings(torch.zeros((1, 4, config.embedding_dim + 1)))

    def test_token_logits_validates_hidden_states(self) -> None:
        config = build_gpt_config(preset="tiny", vocab_size=8, context_length=4)
        model = CausalTextModel(config)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.token_logits(torch.zeros((4, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model.token_logits(torch.zeros((1, 4, config.embedding_dim), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            model.token_logits(torch.zeros((1, 4, config.embedding_dim + 1)))


if __name__ == "__main__":
    unittest.main()
