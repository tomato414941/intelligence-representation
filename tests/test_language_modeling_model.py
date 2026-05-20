from __future__ import annotations

import unittest

import torch

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.models.language_modeling import LanguageModelingModel, build_language_modeling_config
from intrep.text.output_layer import TokenOutputHead


class LanguageModelingModelConfigTest(unittest.TestCase):
    def test_default_shape_matches_project_training_size(self) -> None:
        config = build_language_modeling_config(vocab_size=256, context_length=64)

        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.hidden_dim, 1024)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.dropout, 0.0)

    def test_explicit_lightweight_shape(self) -> None:
        config = _small_config(vocab_size=256, context_length=32)

        self.assertEqual(config.context_length, 32)
        self.assertEqual(config.embedding_dim, 8)
        self.assertEqual(config.num_heads, 2)
        self.assertEqual(config.hidden_dim, 16)

    def test_uses_explicit_model_shape(self) -> None:
        config = build_language_modeling_config(
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
        with self.assertRaisesRegex(ValueError, "embedding_dim must be positive"):
            build_language_modeling_config(vocab_size=256, context_length=8, embedding_dim=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            build_language_modeling_config(vocab_size=256, context_length=8, dropout=1.0)
        with self.assertRaisesRegex(ValueError, "embedding_dim must be divisible by num_heads"):
            build_language_modeling_config(
                vocab_size=256,
                context_length=8,
                embedding_dim=10,
                num_heads=3,
            )

    def test_forward_validates_token_ids(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)

        with self.assertRaisesRegex(ValueError, "rank-2"):
            model(torch.tensor([1, 2], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "context_length"):
            model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "vocabulary range"):
            model(torch.tensor([[1, 8]], dtype=torch.long))

    def test_model_exposes_input_embedding_sequence_path(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)
        token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        embeddings = model.embed_tokens(token_ids)
        encoded = model.encode_embeddings(embeddings)

        self.assertEqual(embeddings.shape, torch.Size([1, 4, config.embedding_dim]))
        self.assertEqual(encoded.shape, torch.Size([1, 4, config.embedding_dim]))

    def test_embed_tokens_supports_position_offset(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)
        token_ids = torch.tensor([[1, 2]], dtype=torch.long)

        offset_embeddings = model.embed_tokens(token_ids, position_offset=2)
        manual_embeddings = model.token_input.token_embedding(token_ids) + model.token_input.position_embedding(
            torch.tensor([[2, 3]], dtype=torch.long)
        )

        self.assertTrue(torch.allclose(offset_embeddings, manual_embeddings))
        with self.assertRaisesRegex(ValueError, "position_offset"):
            model.embed_tokens(token_ids, position_offset=-1)
        with self.assertRaisesRegex(ValueError, "context_length"):
            model.embed_tokens(token_ids, position_offset=3)

    def test_model_uses_shared_transformer_core(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)

        self.assertIsInstance(model.core, SharedTransformerCore)

    def test_model_exposes_token_output_head(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)
        hidden = torch.zeros((1, 4, config.embedding_dim), dtype=torch.float32)

        logits = model.token_logits(hidden)

        self.assertIsInstance(model.token_output, TokenOutputHead)
        self.assertEqual(logits.shape, torch.Size([1, 4, config.vocab_size]))

    def test_encode_embeddings_validates_input_embedding_sequence_shape(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.encode_embeddings(torch.zeros((4, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model.encode_embeddings(torch.zeros((1, 4, config.embedding_dim), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "context_length"):
            model.encode_embeddings(torch.zeros((1, 5, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            model.encode_embeddings(torch.zeros((1, 4, config.embedding_dim + 1)))

    def test_token_logits_validates_hidden_states(self) -> None:
        config = _small_config(vocab_size=8, context_length=4)
        model = LanguageModelingModel(config)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.token_logits(torch.zeros((4, config.embedding_dim)))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model.token_logits(torch.zeros((1, 4, config.embedding_dim), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            model.token_logits(torch.zeros((1, 4, config.embedding_dim + 1)))


def _small_config(*, vocab_size: int, context_length: int):
    return build_language_modeling_config(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=1,
    )


if __name__ == "__main__":
    unittest.main()
