import unittest

import torch

from intrep.fashion_mnist_vit import ImagePatchInputLayer
from intrep.gpt_model import CausalTextModel, build_gpt_config
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.token_scoring import next_token_loss
from intrep.transformer_core import SharedTransformerCore


class ModelInputBoundariesTest(unittest.TestCase):
    def test_text_and_image_embeddings_can_use_same_transformer_core(self) -> None:
        embedding_dim = 8
        core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )
        text_model = CausalTextModel(
            build_gpt_config(
                preset="tiny",
                vocab_size=16,
                context_length=8,
                embedding_dim=embedding_dim,
            )
        )
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=embedding_dim,
        )

        text_embeddings = text_model.embed_tokens(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        image_embeddings = image_input(torch.zeros((1, 4, 4), dtype=torch.float32))
        text_hidden = core(text_embeddings)
        image_hidden = core(image_embeddings)

        self.assertEqual(text_embeddings.shape, torch.Size([1, 4, embedding_dim]))
        self.assertEqual(image_embeddings.shape, torch.Size([1, 4, embedding_dim]))
        self.assertEqual(text_hidden.shape, torch.Size([1, 4, embedding_dim]))
        self.assertEqual(image_hidden.shape, torch.Size([1, 4, embedding_dim]))

    def test_shared_core_preserves_gradient_path_for_model_input_embeddings(self) -> None:
        embedding_dim = 8
        core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )
        embeddings = torch.randn((2, 3, embedding_dim), dtype=torch.float32, requires_grad=True)

        loss = core(embeddings).sum()
        loss.backward()

        self.assertIsNotNone(embeddings.grad)
        self.assertEqual(embeddings.grad.shape, embeddings.shape)

    def test_shared_core_supports_causal_encoding(self) -> None:
        core = SharedTransformerCore(
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )
        embeddings = torch.zeros((1, 4, 8), dtype=torch.float32)

        hidden = core(embeddings, causal=True)

        self.assertEqual(hidden.shape, torch.Size([1, 4, 8]))

    def test_image_and_text_embeddings_can_be_concatenated_for_one_core_pass(self) -> None:
        embedding_dim = 8
        core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )
        text_model = CausalTextModel(
            build_gpt_config(
                preset="tiny",
                vocab_size=16,
                context_length=8,
                embedding_dim=embedding_dim,
            )
        )
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=embedding_dim,
        )

        image_embeddings = image_input(torch.zeros((1, 4, 4), dtype=torch.float32))
        text_embeddings = text_model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long))
        combined = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
        hidden = core(combined)

        self.assertEqual(image_embeddings.shape, torch.Size([1, 4, embedding_dim]))
        self.assertEqual(text_embeddings.shape, torch.Size([1, 2, embedding_dim]))
        self.assertEqual(combined.shape, torch.Size([1, 6, embedding_dim]))
        self.assertEqual(hidden.shape, torch.Size([1, 6, embedding_dim]))
        self.assertTrue(torch.allclose(combined[:, :4, :], image_embeddings))
        self.assertTrue(torch.allclose(combined[:, 4:, :], text_embeddings))

    def test_image_and_text_embeddings_can_produce_token_logits(self) -> None:
        embedding_dim = 8
        vocab_size = 16
        text_model = CausalTextModel(
            build_gpt_config(
                preset="tiny",
                vocab_size=vocab_size,
                context_length=8,
                embedding_dim=embedding_dim,
            )
        )
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=embedding_dim,
        )

        image_embeddings = image_input(torch.zeros((1, 4, 4), dtype=torch.float32))
        prompt_embeddings = text_model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long))
        combined = concatenate_input_embedding_sequences(image_embeddings, prompt_embeddings)
        hidden = text_model.encode_embeddings(combined, causal=True)
        logits = text_model.token_logits(hidden)

        self.assertEqual(combined.shape, torch.Size([1, 6, embedding_dim]))
        self.assertEqual(hidden.shape, torch.Size([1, 6, embedding_dim]))
        self.assertEqual(logits.shape, torch.Size([1, 6, vocab_size]))

    def test_image_conditioned_text_logits_can_be_scored_with_loss_mask(self) -> None:
        embedding_dim = 8
        vocab_size = 16
        text_model = CausalTextModel(
            build_gpt_config(
                preset="tiny",
                vocab_size=vocab_size,
                context_length=8,
                embedding_dim=embedding_dim,
            )
        )
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=embedding_dim,
        )

        image_embeddings = image_input(torch.zeros((1, 4, 4), dtype=torch.float32))
        text_token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        text_embeddings = text_model.embed_tokens(text_token_ids)
        combined = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
        hidden = text_model.encode_embeddings(combined, causal=True)
        logits = text_model.token_logits(hidden)
        target_token_ids = torch.tensor([[0, 0, 0, 0, 1, 2, 3]], dtype=torch.long)
        loss_mask = torch.tensor([[False, False, False, False, False, False, True]], dtype=torch.bool)

        loss = next_token_loss(logits, target_token_ids, loss_mask=loss_mask)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_concatenate_input_embedding_sequences_validates_boundaries(self) -> None:
        embeddings = torch.zeros((1, 2, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "at least one"):
            concatenate_input_embedding_sequences()
        with self.assertRaisesRegex(ValueError, "shape"):
            concatenate_input_embedding_sequences(torch.zeros((2, 8), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "floating point"):
            concatenate_input_embedding_sequences(torch.zeros((1, 2, 8), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "batch size"):
            concatenate_input_embedding_sequences(
                embeddings,
                torch.zeros((2, 2, 8), dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "hidden size"):
            concatenate_input_embedding_sequences(
                embeddings,
                torch.zeros((1, 2, 9), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
