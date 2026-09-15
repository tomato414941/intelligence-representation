from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.evaluation import evaluate_panel, make_panel
from intrep.problems.shared_prediction.questions import _metric_means
from intrep.problems.shared_prediction.rule_transfer_training import state_digest
from intrep.problems.shared_prediction.sources import build_sources
from tests.test_shared_prediction_questions import question_recipe
from tests.test_shared_prediction_sources import make_model, make_tokenizer


class SourceBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import intrep.problems.shared_prediction.record_sources  # noqa: F401
        torch.set_num_threads(1)

    def fixture(self, root):
        recipe = question_recipe(root)
        entries = []
        for index, length in enumerate((8, 8, 20)):
            path = root / f"speech-{index}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes((np.arange(length, dtype=np.int16) * (index + 1)).tobytes())
            entries.append({"path": path.name, "label": index % 2, "speaker": "train", "split": "train",
                            "sample_rate": 8000, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (root / "speech.json").write_text(json.dumps({"records": entries}))
        np.savez(root / "sensors.npz", signals=np.random.default_rng(7).normal(size=(3, 128, 9)).astype(np.float32),
                 labels=np.arange(3), subjects=np.zeros(3, dtype=np.int64))
        (root / "normalization.json").write_text(json.dumps({"mean": [0] * 9, "std": [1] * 9}))
        recipe["sources"].extend([
            {"name": "spoken_digits", "kind": "spoken_digits", "manifest": "speech.json", "split": "train"},
            {"name": "inertial_activity", "kind": "inertial_activity", "path": "sensors.npz", "normalization": "normalization.json"},
        ])
        model = make_model()
        return model, build_sources(model, make_tokenizer(), recipe, root)

    def test_collected_metrics_preserve_precision_and_means_for_missing_keys(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                rows = [
                    {"accuracy": torch.tensor(.25, device=device, requires_grad=True),
                     "precise": torch.tensor(1 + 2 ** -40, dtype=torch.float64, device=device),
                     "integer": torch.tensor(2 ** 55 + 1, device=device),
                     "optional": torch.tensor(3, dtype=torch.bfloat16, device=device)},
                    {"accuracy": .75, "precise": 1.25, "integer": torch.tensor(3, device=device)},
                    {"accuracy": torch.tensor(.5, dtype=torch.float16, device=device), "optional": 5.0},
                ]
                expected = {"accuracy": .5, "precise": (1 + 2 ** -40 + 1.25) / 2,
                            "integer": (float(2 ** 55 + 1) + 3) / 2, "optional": 4.0}
                actual = _metric_means(rows)
                self.assertEqual(actual, expected)
                self.assertTrue(all(type(value) is float for value in actual.values()))
                self.assertEqual(json.loads(json.dumps(actual)), expected)

    def test_nonfinite_question_metrics_are_rejected_before_parameter_updates(self):
        for form in ("original", "identify"):
            with self.subTest(form=form), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe = question_recipe(root)
                recipe["sources"][1]["records_per_update"] = 4
                model = make_model()
                source = build_sources(model, make_tokenizer(), recipe, root)["mnist"]
                source.step = 0 if form == "original" else 1
                target = source.reader if form == "original" else source
                method = "record_batch_loss" if form == "original" else "_score"
                score = getattr(target, method)
                calls = []

                def invalid_metrics(record):
                    loss = score(record)
                    calls.append(True)
                    invalid = form == "original" or len(calls) == 2
                    target.last_metrics["invalid"] = loss.detach().new_tensor(float("nan") if invalid else 1)
                    return loss

                setattr(target, method, invalid_metrics)
                trainer = JointTrainer(model, {"mnist": 1}, learning_rate=.01)
                before = copy.deepcopy(model.state_dict())
                with self.assertRaisesRegex(ValueError, "nonfinite metric"):
                    trainer.step({"mnist": source.loss})
                self.assertEqual(trainer.steps, 0)
                self.assertFalse(trainer.optimizer.state)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_batching_preserves_per_example_losses_gradients_and_all_sequence_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            model, sources = self.fixture(Path(directory))
            model.eval()
            for name, source in sources.items():
                with self.subTest(source=name):
                    reader = source.reader
                    records = ([reader.next_record() for _ in range(3)] if name == "text_data"
                               else [reader.read_record(index) for index in range(3)])
                    model.zero_grad(set_to_none=True)
                    expected = torch.stack([reader.record_loss(record) for record in records]).mean()
                    expected.backward()
                    gradients = {key: parameter.grad.clone() for key, parameter in model.named_parameters()
                                 if parameter.grad is not None}
                    model.zero_grad(set_to_none=True)
                    calls = []
                    handle = model.core.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
                    actual = reader.record_batch_loss(records)
                    handle.remove()
                    actual.backward()
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    for key, gradient in gradients.items():
                        torch.testing.assert_close(dict(model.named_parameters())[key].grad, gradient, rtol=1e-4, atol=1e-6)
                    self.assertEqual(sum(shape[0] for shape in calls), len(records))
                    if name == "spoken_digits":
                        self.assertEqual([(shape[0], shape[1]) for shape in calls], [(2, 2), (1, 5)])
                    else:
                        self.assertEqual(len(calls), 1)

    def test_larger_pair_batches_are_consumed_and_evaluation_remains_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root)
            recipe["sources"][1]["records_per_update"] = 8
            model = make_model()
            source = build_sources(model, make_tokenizer(), recipe, root)["mnist"]
            calls = []
            handle = model.core.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
            source.loss().backward()
            handle.remove()
            self.assertEqual(calls, [8])
            self.assertEqual(source.last_update_info, {"form": "original", "records": 8, "questions": 8})
            self.assertEqual(source.reader.sampler.samples, 4)
            self.assertEqual(set(source.last_response["record_indices"]), set(range(4)))
            state = copy.deepcopy(source.state_dict())
            expected = source.loss().detach()
            self.assertEqual(source.last_update_info, {"form": "identify", "records": 8, "questions": 4})
            self.assertEqual(len(source.last_response["responses"]), 4)
            source.load_state_dict(state)
            panel = make_panel({"mnist": source}, 2)
            for row in panel["mnist"]:
                row["generate"] = False
            evaluate_panel(model, {"mnist": source}, panel)
            self.assertEqual(state_digest(source.state_dict()), state_digest(state))
            torch.testing.assert_close(source.loss().detach(), expected, rtol=0, atol=0)

    def test_incomplete_pair_counts_are_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root)
            for count in (0, 1, 3, True, 2.5):
                with self.subTest(count=count):
                    recipe["sources"][1]["records_per_update"] = count
                    with self.assertRaisesRegex(ValueError, "even count"):
                        build_sources(make_model(), make_tokenizer(), recipe, root)


if __name__ == "__main__":
    unittest.main()
