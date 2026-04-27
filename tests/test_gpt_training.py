import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.gpt_training import GPTTrainingConfig, language_model_batches, train_mixed_gpt
from intrep.mixed_corpus import MixedDocument, default_mixed_documents, render_corpus


class GPTTrainingTest(unittest.TestCase):
    def test_language_model_batches_shift_targets_by_one_token(self) -> None:
        token_ids = list(range(20))

        inputs, targets = language_model_batches(token_ids, context_length=4, batch_size=2)

        self.assertEqual(inputs.shape, torch.Size([2, 2, 4]))
        self.assertEqual(targets.shape, torch.Size([2, 2, 4]))
        self.assertEqual(inputs[0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(targets[0, 0].tolist(), [1, 2, 3, 4])

    def test_decoder_only_gpt_forward_returns_token_logits(self) -> None:
        model = DecoderOnlyGPT(
            GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            )
        )
        token_ids = torch.zeros((2, 8), dtype=torch.long)

        logits = model(token_ids)

        self.assertEqual(logits.shape, torch.Size([2, 8, ByteTokenizer.vocab_size]))

    def test_training_runs_on_mixed_corpus_and_reduces_loss(self) -> None:
        result = train_mixed_gpt(
            documents=default_mixed_documents(),
            training_config=GPTTrainingConfig(
                context_length=32,
                batch_size=4,
                max_steps=12,
                learning_rate=0.01,
                seed=11,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=32,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )

        self.assertEqual(result.steps, 12)
        self.assertEqual(len(result.loss_history), 12)
        self.assertGreater(result.token_count, len(render_corpus(default_mixed_documents())))
        self.assertEqual(result.initial_loss, result.loss_history[0])
        self.assertEqual(result.final_loss, result.loss_history[-1])
        self.assertEqual(result.best_loss, min(result.loss_history))
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertEqual(result.loss_reduction, result.initial_loss - result.final_loss)
        self.assertIsNone(result.initial_eval_loss)
        self.assertIsNone(result.final_eval_loss)

    def test_training_reports_held_out_eval_loss(self) -> None:
        train_documents = [
            MixedDocument(
                id="train_tiny_001",
                modality="text",
                content="alpha beta gamma alpha beta gamma alpha beta gamma",
            )
        ]
        eval_documents = [
            MixedDocument(
                id="eval_tiny_001",
                modality="text",
                content="delta epsilon zeta delta epsilon zeta delta epsilon zeta",
            )
        ]

        result = train_mixed_gpt(
            documents=train_documents,
            eval_documents=eval_documents,
            training_config=GPTTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=3,
                learning_rate=0.005,
                seed=13,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )

        self.assertIsNotNone(result.initial_eval_loss)
        self.assertIsNotNone(result.final_eval_loss)
        self.assertGreater(result.initial_eval_loss, 0.0)
        self.assertGreater(result.final_eval_loss, 0.0)
        self.assertNotEqual(result.initial_eval_loss, result.final_eval_loss)

    def test_rejects_empty_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "documents must not be empty"):
            train_mixed_gpt(documents=[])

    def test_rejects_empty_eval_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_documents must not be empty"):
            train_mixed_gpt(documents=default_mixed_documents(), eval_documents=[])

    def test_rejects_invalid_training_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            train_mixed_gpt(training_config=GPTTrainingConfig(batch_size=0))

    def test_rejects_model_config_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length must match"):
            train_mixed_gpt(
                training_config=GPTTrainingConfig(context_length=16),
                model_config=GPTConfig(vocab_size=ByteTokenizer.vocab_size, context_length=8),
            )


if __name__ == "__main__":
    unittest.main()
