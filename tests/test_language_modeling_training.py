import tempfile
import unittest
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, CausalTextConfig
from intrep.language_modeling_training import (
    LanguageModelingTrainingArtifacts,
    LanguageModelingTrainingConfig,
    language_model_batches,
    resolve_training_device,
    train_language_modeling_with_artifacts,
    _train_text_corpus_with_artifacts,
)
from intrep.text_examples import LanguageModelingExample


class LanguageModelingTrainingTest(unittest.TestCase):
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

    def test_language_model_batches_validates_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length must be positive"):
            language_model_batches([1, 2, 3], context_length=0, batch_size=1)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            language_model_batches([1, 2, 3], context_length=1, batch_size=0)

    def test_language_model_batches_logs_window_summary(self) -> None:
        with self.assertLogs("intrep.language_modeling_training", level="DEBUG") as logs:
            language_model_batches(list(range(20)), context_length=4, batch_size=3)

        self.assertIn("window_count=", logs.output[0])
        self.assertIn("dropped_window_count=", logs.output[0])

    def test_causal_text_model_forward_returns_token_logits(self) -> None:
        model = CausalTextModel(
            CausalTextConfig(
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

    def test_training_runs_on_text_corpus_and_reduces_loss(self) -> None:
        artifacts = _train_text_corpus_with_artifacts(
            corpus="alpha beta gamma alpha beta gamma alpha beta gamma " * 4,
            training_config=LanguageModelingTrainingConfig(
                context_length=32,
                batch_size=4,
                max_steps=12,
                learning_rate=0.01,
                seed=11,
            ),
            model_config=CausalTextConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=32,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )
        result = artifacts.result

        self.assertEqual(result.steps, 12)
        self.assertEqual(len(result.loss_history), 12)
        self.assertGreater(result.token_count, 0)
        self.assertEqual(result.initial_loss, result.loss_history[0])
        self.assertEqual(result.final_loss, result.loss_history[-1])
        self.assertEqual(result.initial_step_loss, result.initial_loss)
        self.assertEqual(result.final_step_loss, result.final_loss)
        self.assertEqual(result.best_loss, min(result.loss_history))
        self.assertEqual(result.best_step_loss, result.best_loss)
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertEqual(result.loss_reduction, result.initial_loss - result.final_loss)
        self.assertEqual(result.step_loss_reduction, result.loss_reduction)
        self.assertEqual(result.step_loss_reduction_ratio, result.loss_reduction_ratio)
        self.assertIsNotNone(result.initial_train_loss)
        self.assertIsNotNone(result.final_train_loss)
        self.assertGreater(result.initial_train_loss, 0.0)
        self.assertGreater(result.final_train_loss, 0.0)
        self.assertIsNone(result.initial_eval_loss)
        self.assertIsNone(result.final_eval_loss)
        self.assertEqual(result.eval_split, "train")
        self.assertFalse(result.generalization_eval)
        self.assertTrue(result.warnings)
        self.assertEqual(result.device, "cpu")

    def test_training_runs_on_language_modeling_examples(self) -> None:
        artifacts = train_language_modeling_with_artifacts(
            train_examples=(
                LanguageModelingExample("alpha beta gamma alpha beta gamma"),
                LanguageModelingExample("alpha beta gamma alpha beta gamma"),
            ),
            eval_examples=(LanguageModelingExample("delta epsilon zeta delta epsilon zeta"),),
            training_config=LanguageModelingTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=2,
                learning_rate=0.005,
                seed=23,
            ),
            model_config=CausalTextConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )

        self.assertIsInstance(artifacts, LanguageModelingTrainingArtifacts)
        self.assertEqual(artifacts.result.eval_split, "held_out")
        self.assertTrue(artifacts.result.generalization_eval)
        self.assertGreater(artifacts.result.token_count, 0)

    def test_resolve_training_device_auto_prefers_cuda_when_available(self) -> None:
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_training_device("auto").type, "cuda")

    def test_rejects_unavailable_cuda_device(self) -> None:
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA device requested"):
                resolve_training_device("cuda")

    def test_training_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints" / "causal-text.pt"
            with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
                artifacts = _train_text_corpus_with_artifacts(
                    corpus="one two three one two three one two three",
                    training_config=LanguageModelingTrainingConfig(
                        context_length=8,
                        batch_size=2,
                        max_steps=2,
                        learning_rate=0.005,
                        seed=19,
                        device="auto",
                        checkpoint_path=checkpoint_path,
                    ),
                    model_config=CausalTextConfig(
                        vocab_size=ByteTokenizer.vocab_size,
                        context_length=8,
                        embedding_dim=16,
                        num_heads=2,
                        hidden_dim=32,
                    ),
                )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["schema_version"], "intrep.causal_text_checkpoint.v1")
        self.assertEqual(payload["model_config"]["context_length"], 8)
        self.assertEqual(payload["training_config"]["device"], "auto")
        self.assertEqual(payload["training_config"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(payload["tokenizer"], {"kind": "byte"})
        self.assertEqual(payload["result"]["device"], "cpu")
        self.assertEqual(artifacts.result.device, "cpu")
        self.assertTrue(payload["model_state_dict"])
        self.assertTrue(
            all(tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values())
        )

    def test_training_writes_byte_pair_tokenizer_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints" / "causal-text.pt"
            _train_text_corpus_with_artifacts(
                corpus="hello hello hello hello",
                training_config=LanguageModelingTrainingConfig(
                    context_length=4,
                    batch_size=2,
                    max_steps=1,
                    learning_rate=0.005,
                    seed=19,
                    checkpoint_path=checkpoint_path,
                    tokenizer="byte-pair",
                    tokenizer_vocab_size=260,
                ),
                model_config=CausalTextConfig(
                    vocab_size=260,
                    context_length=4,
                    embedding_dim=16,
                    num_heads=2,
                    hidden_dim=32,
                ),
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["tokenizer"]["kind"], "byte-pair")
        self.assertEqual(payload["tokenizer"]["vocab_size"], 260)
        self.assertTrue(payload["tokenizer"]["merges"])

    def test_training_reports_held_out_eval_loss(self) -> None:
        artifacts = _train_text_corpus_with_artifacts(
            corpus="alpha beta gamma alpha beta gamma alpha beta gamma",
            eval_corpus="delta epsilon zeta delta epsilon zeta delta epsilon zeta",
            training_config=LanguageModelingTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=3,
                learning_rate=0.005,
                seed=13,
            ),
            model_config=CausalTextConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )
        result = artifacts.result

        self.assertIsNotNone(result.initial_eval_loss)
        self.assertIsNotNone(result.final_eval_loss)
        self.assertGreater(result.initial_eval_loss, 0.0)
        self.assertGreater(result.final_eval_loss, 0.0)
        self.assertNotEqual(result.initial_eval_loss, result.final_eval_loss)
        self.assertEqual(result.eval_split, "held_out")
        self.assertTrue(result.generalization_eval)
        self.assertEqual(result.warnings, ())

    def test_training_with_artifacts_returns_model_tokenizer_and_result(self) -> None:
        artifacts = _train_text_corpus_with_artifacts(
            corpus="red green blue red green blue red green blue",
            training_config=LanguageModelingTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=2,
                learning_rate=0.005,
                seed=17,
            ),
            model_config=CausalTextConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            ),
        )

        self.assertIsInstance(artifacts, LanguageModelingTrainingArtifacts)
        self.assertIsInstance(artifacts.result.steps, int)
        self.assertEqual(artifacts.result.steps, 2)
        self.assertIsInstance(artifacts.model, CausalTextModel)
        self.assertIsInstance(artifacts.tokenizer, ByteTokenizer)

        token_ids = artifacts.tokenizer.encode("red blue")[:8]
        token_ids.extend([0] * (8 - len(token_ids)))
        inputs = torch.tensor([token_ids], dtype=torch.long)

        artifacts.model.eval()
        with torch.no_grad():
            logits = artifacts.model(inputs)

        self.assertEqual(logits.shape, torch.Size([1, 8, artifacts.tokenizer.vocab_size]))

    def test_training_can_continue_from_initial_model(self) -> None:
        initial_model = CausalTextModel(
            CausalTextConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
            )
        )

        artifacts = _train_text_corpus_with_artifacts(
            corpus="red green blue red green blue red green blue",
            training_config=LanguageModelingTrainingConfig(
                context_length=8,
                batch_size=2,
                max_steps=1,
                learning_rate=0.005,
                seed=17,
            ),
            model_config=initial_model.config,
            initial_model=initial_model,
        )

        self.assertIs(artifacts.model, initial_model)
        self.assertEqual(artifacts.model.config, initial_model.config)

    def test_rejects_empty_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "corpus must not be empty"):
            _train_text_corpus_with_artifacts(corpus="")

    def test_rejects_empty_eval_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_corpus must not be empty"):
            _train_text_corpus_with_artifacts(
                corpus="abc abc abc abc abc",
                eval_corpus="",
                training_config=LanguageModelingTrainingConfig(context_length=4),
            )

    def test_rejects_invalid_training_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            _train_text_corpus_with_artifacts(
                corpus="abc",
                training_config=LanguageModelingTrainingConfig(batch_size=0),
            )
        with self.assertRaisesRegex(ValueError, "batch_stride must be positive"):
            _train_text_corpus_with_artifacts(
                corpus="abc",
                training_config=LanguageModelingTrainingConfig(batch_stride=0),
            )

    def test_rejects_model_config_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length must match"):
            _train_text_corpus_with_artifacts(
                corpus="abc abc abc abc abc",
                training_config=LanguageModelingTrainingConfig(context_length=16),
                model_config=CausalTextConfig(vocab_size=ByteTokenizer.vocab_size, context_length=8),
            )


if __name__ == "__main__":
    unittest.main()
