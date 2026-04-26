import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.gpt_training import GPTTrainingConfig, language_model_batches, train_mixed_gpt
from intrep.mixed_corpus import default_mixed_documents, render_corpus


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
        self.assertGreater(result.token_count, len(render_corpus(default_mixed_documents())))
        self.assertLess(result.final_loss, result.initial_loss)


if __name__ == "__main__":
    unittest.main()
