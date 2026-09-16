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
from intrep.problems.shared_prediction.questions import Prediction, Question, _metric_means, waveform_question
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
                method = "record_batch_loss" if form == "original" else "_score_batch"
                score = getattr(target, method)

                def invalid_metrics(record):
                    result = score(record)
                    if form == "original":
                        target.last_metrics["invalid"] = result.detach().new_tensor(float("nan"))
                    else:
                        for index, (loss, metrics, _) in enumerate(result):
                            metrics["invalid"] = loss.detach().new_tensor(float("nan") if index == 1 else 1)
                    return result

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

    def assert_question_batch_matches_singles(self, source, questions):
        model = source.model
        model.zero_grad(set_to_none=True)
        expected = []
        for question in questions:
            loss = source._score(question)
            expected.append((loss, source.last_metrics.copy(), copy.deepcopy(source.last_response)))
        torch.stack([row[0] for row in expected]).mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        calls = []
        handle = model.core.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
        actual = source._score_batch(questions)
        handle.remove()
        torch.stack([row[0] for row in actual]).mean().backward()
        self.assertEqual({name for name, parameter in model.named_parameters() if parameter.grad is not None},
                         set(gradients))
        for name, gradient in gradients.items():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, gradient, rtol=1e-4, atol=1e-6)
        for (loss, metrics, response), (old_loss, old_metrics, old_response) in zip(actual, expected):
            torch.testing.assert_close(loss, old_loss, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(metrics, old_metrics, rtol=1e-5, atol=1e-6)
            self.assertEqual(response, old_response)
        return calls

    def test_answer_batches_keep_question_weights_and_order_with_different_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            _, sources = self.fixture(Path(directory))
            source = sources["text_data"]
            questions = [Question("one", "two"), Question("one two three", "four go left"),
                         Question("one", "left"), Question("one", ""), Question("one two", "red")]
            calls = self.assert_question_batch_matches_singles(source, questions)
            self.assertEqual([shape[0] for shape in calls], [2, 1, 1, 1])
            # Generation starts from each observation/question prefix, including
            # when teacher-forced answers of different lengths share a batch.
            source._forced = {"generate": True}
            self.assert_question_batch_matches_singles(source, questions)

    def test_audio_batches_preserve_partial_chunk_masks_and_per_question_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            _, sources = self.fixture(Path(directory))
            source = sources["spoken_digits"]
            questions = [waveform_question(source, {"audio": torch.linspace(.2, .8, length), "sample_rate": 8000}, 0)
                         for length in (1, 8, 3)]
            self.assertEqual([int(question.predictions[0].valid.sum()) for question in questions], [1, 4, 3])
            calls = self.assert_question_batch_matches_singles(source, questions)
            self.assertEqual([shape[0] for shape in calls], [2, 1])
            altered = copy.deepcopy(questions)
            for question in altered:
                prediction = question.predictions[0]
                prediction.target[~prediction.valid] = 1000
                prediction.baseline[~prediction.valid] = -1000
            for before, after in zip(source._score_batch(questions), source._score_batch(altered)):
                torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
                torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)

    def test_prediction_batches_keep_multiple_head_weights_and_optional_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            model, sources = self.fixture(Path(directory))
            source = sources["mnist"]
            model.attach_output("board_pieces", torch.nn.Linear(model.dimension, 29))
            model.attach_output("board_hands", torch.nn.Linear(model.dimension, 19))
            questions = []
            for prompt, occupied in (("one", True), ("one two three", True), ("one", False)):
                pieces = torch.arange(81) % 29 if occupied else torch.zeros(81, dtype=torch.long)
                hands = torch.arange(14)
                questions.append(Question(prompt, predictions=[
                    Prediction("board_pieces", torch.arange(81).view(-1, 1), pieces, "class", torch.zeros_like(pieces)),
                    Prediction("board_hands", torch.arange(14).view(-1, 1), hands, "class", torch.zeros_like(hands)),
                ]))
            # Different output layouts must not share head offsets, even when
            # the total sequence lengths happen to match.
            questions.insert(1, Question("one", "two"))
            questions.append(Question("one", predictions=list(reversed(questions[0].predictions))))
            calls = self.assert_question_batch_matches_singles(source, questions)
            self.assertEqual(sorted(shape[0] for shape in calls), [1, 1, 1, 2])
            metrics = [row[1] for row in source._score_batch(questions)]
            self.assertIn("occupied_square_accuracy", metrics[0])
            self.assertNotIn("occupied_square_accuracy", metrics[3])

    def test_all_added_forms_preserve_sampling_counts_responses_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            model, sources = self.fixture(Path(directory))
            grouped_updates = 0
            for source in sources.values():
                source.records_per_update = 8
                batch_score = source._score_batch
                for step in range(2 * (len(source.forms) - 1)):
                    with self.subTest(source=source.config["name"], step=step):
                        state, rng = copy.deepcopy(source.state_dict()), torch.get_rng_state().clone()
                        source._score_batch = lambda questions: [batch_score([question])[0] for question in questions]
                        model.zero_grad(set_to_none=True)
                        expected = source.loss()
                        expected.backward()
                        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                                     if parameter.grad is not None}
                        expected_state = state_digest(source.state_dict())
                        metrics, response, info = copy.deepcopy((source.last_metrics, source.last_response, source.last_update_info))
                        expected_rng = torch.get_rng_state().clone()
                        source.load_state_dict(state)
                        torch.set_rng_state(rng)
                        source._score_batch = batch_score
                        model.zero_grad(set_to_none=True)
                        calls = []
                        handle = model.core.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
                        actual = source.loss()
                        handle.remove()
                        actual.backward()
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        self.assertEqual({name for name, parameter in model.named_parameters() if parameter.grad is not None},
                                         set(gradients))
                        for name, gradient in gradients.items():
                            torch.testing.assert_close(dict(model.named_parameters())[name].grad, gradient, rtol=1e-4, atol=1e-6)
                        torch.testing.assert_close(source.last_metrics, metrics, rtol=1e-5, atol=1e-6)
                        self.assertEqual((source.last_response, source.last_update_info), (response, info))
                        self.assertEqual(state_digest(source.state_dict()), expected_state)
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                        self.assertEqual(sum(shape[0] for shape in calls), info["questions"])
                        if source.last_form != "original" and len(calls) < info["questions"]:
                            grouped_updates += 1
            self.assertGreater(grouped_updates, 10)

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
