import tempfile
import unittest
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.gpt_training import (
    GPTTrainingArtifacts,
    GPTTrainingConfig,
    language_model_batches,
    resolve_training_device,
    train_mixed_gpt,
    train_mixed_gpt_with_artifacts,
)
from intrep.mixed_corpus import MixedDocument, default_mixed_documents, render_corpus


class GPTTrainingTest(unittest.TestCase):
    def test_language_model_batches_shift_targets_by_one_token(self) -> None:
        token_ids = list(range(20))

        inputs, targets = language_model_batches(token_ids, context_length=4, batch_size=2)

        self.assertEqual(inputs.shape, torch.Size([2, 2, 4]))
        self.assertEqual(targets.shape, torch.Size([2, 2, 4]))
        self.assertEqual(inputs[0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(targets[0, 0].tolist(), [1, 2, 3, 4])

    def test_language_model_batches_stride_creates_overlapping_windows(self) -> None:
        token_ids = list(range(12))

        default_inputs, _ = language_model_batches(token_ids, context_length=4, batch_size=2)
        strided_inputs, strided_targets = language_model_batches(
            token_ids,
            context_length=4,
            batch_size=2,
            batch_stride=2,
        )

        self.assertEqual(default_inputs.shape, torch.Size([1, 2, 4]))
        self.assertEqual(strided_inputs.shape, torch.Size([2, 2, 4]))
        self.assertEqual(strided_inputs[0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(strided_inputs[0, 1].tolist(), [2, 3, 4, 5])
        self.assertEqual(strided_targets[0, 1].tolist(), [3, 4, 5, 6])

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
        self.assertIsNotNone(result.initial_train_loss)
        self.assertIsNotNone(result.final_train_loss)
        self.assertGreater(result.initial_train_loss, 0.0)
        self.assertGreater(result.final_train_loss, 0.0)
        self.assertIsNone(result.initial_eval_loss)
        self.assertIsNone(result.final_eval_loss)
        self.assertEqual(result.device, "cpu")

    def test_resolve_training_device_auto_prefers_cuda_when_available(self) -> None:
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_training_device("auto").type, "cuda")

    def test_rejects_unavailable_cuda_device(self) -> None:
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA device requested"):
                resolve_training_device("cuda")

    def test_training_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints" / "gpt.pt"
            with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
                artifacts = train_mixed_gpt_with_artifacts(
                    documents=[
                        MixedDocument(
                            id="checkpoint_tiny_001",
                            modality="text",
                            content="one two three one two three one two three",
                        )
                    ],
                    training_config=GPTTrainingConfig(
                        context_length=8,
                        batch_size=2,
                        max_steps=2,
                        learning_rate=0.005,
                        seed=19,
                        device="auto",
                        checkpoint_path=checkpoint_path,
                    ),
                    model_config=GPTConfig(
                        vocab_size=ByteTokenizer.vocab_size,
                        context_length=8,
                        embedding_dim=16,
                        num_heads=2,
                        hidden_dim=32,
                    ),
                )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["schema_version"], "intrep.gpt_checkpoint.v1")
        self.assertEqual(payload["model_config"]["context_length"], 8)
        self.assertEqual(payload["training_config"]["device"], "auto")
        self.assertEqual(payload["training_config"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(payload["result"]["device"], "cpu")
        self.assertEqual(artifacts.result.device, "cpu")
        self.assertTrue(payload["model_state_dict"])
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values()))

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

    def test_training_with_artifacts_returns_model_tokenizer_and_result(self) -> None:
        artifacts = train_mixed_gpt_with_artifacts(
            documents=[
                MixedDocument(
                    id="artifact_tiny_001",
                    modality="text",
                    content="red green blue red green blue red green blue",
                )
            ],
            training_config=GPTTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=2,
                learning_rate=0.005,
                seed=17,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )

        self.assertIsInstance(artifacts, GPTTrainingArtifacts)
        self.assertIsInstance(artifacts.result.steps, int)
        self.assertEqual(artifacts.result.steps, 2)
        self.assertIsInstance(artifacts.model, DecoderOnlyGPT)
        self.assertIsInstance(artifacts.tokenizer, ByteTokenizer)

        token_ids = artifacts.tokenizer.encode("red blue")[:8]
        token_ids.extend([0] * (8 - len(token_ids)))
        inputs = torch.tensor([token_ids], dtype=torch.long)

        artifacts.model.eval()
        with torch.no_grad():
            logits = artifacts.model(inputs)

        self.assertEqual(logits.shape, torch.Size([1, 8, artifacts.tokenizer.vocab_size]))

    def test_rejects_empty_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "documents must not be empty"):
            train_mixed_gpt(documents=[])

    def test_rejects_empty_eval_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_documents must not be empty"):
            train_mixed_gpt(documents=default_mixed_documents(), eval_documents=[])

    def test_rejects_invalid_training_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            train_mixed_gpt(training_config=GPTTrainingConfig(batch_size=0))
        with self.assertRaisesRegex(ValueError, "batch_stride must be positive"):
            train_mixed_gpt(training_config=GPTTrainingConfig(batch_stride=0))

    def test_rejects_model_config_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length must match"):
            train_mixed_gpt(
                training_config=GPTTrainingConfig(context_length=16),
                model_config=GPTConfig(vocab_size=ByteTokenizer.vocab_size, context_length=8),
            )


if __name__ == "__main__":
    unittest.main()
