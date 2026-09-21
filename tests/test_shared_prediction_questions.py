from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import shogi
import torch

from intrep.experience.multimodal.records import MultimodalEpisode
from intrep.problems.shared_prediction.evaluation import (
    evaluate_panel,
    make_panel,
    set_case,
)
from intrep.problems.shared_prediction.questions import (
    board_labels,
    categorical_question,
    history_question,
    image_question,
    sensor_question,
    shogi_question,
    waveform_question,
)
from intrep.problems.shared_prediction.sources import NativeSource, build_sources
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from tests.test_shared_prediction_sources import make_model, make_recipe, make_tokenizer


def question_recipe(root, mode="varied"):
    recipe = make_recipe(root)
    recipe["defaults"].update(question_mode=mode, records_per_update=2)
    recipe["sources"][0]["records_per_update"] = 1
    recipe["sources"][1]["name"] = "mnist"
    return recipe


class QuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import transformers  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("install the lfm extra for question integration tests") from error
        torch.set_num_threads(1)

    def test_question_only_change_reverses_answer_on_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sources(make_model(), make_tokenizer(), question_recipe(root), root)["mnist"]
            case = next(row for row in make_panel({"mnist": source}, 3)["mnist"] if row["form"] == "same")
            set_case(source, {**case, "generate": False})
            first = source._records(case["seed"])
            source.step += 19
            set_case(source, {**case, "form": "different", "generate": False})
            second = source._records(case["seed"])
            self.assertEqual([row["index"] for row in first], [row["index"] for row in second])
            same = categorical_question(source.reader, first, "same", 0)
            different = categorical_question(source.reader, second, "different", 1)
            self.assertNotEqual(same.answer, different.answer)
            for a, b in zip(first, second):
                torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)

    def test_hidden_image_and_audio_targets_do_not_change_inputs_or_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sources(make_model(), make_tokenizer(), question_recipe(root), root)["mnist"]
            record = {"image": torch.rand(4, 4, 3) + 0.1}
            before = image_question(source.reader, record, 12, 0)
            hidden = before.inputs[0][1][0] == 0
            record["image"][hidden] += 1
            after = image_question(source.reader, record, 12, 0)
            torch.testing.assert_close(before.inputs[0][1][0], after.inputs[0][1][0], rtol=0, atol=0)
            torch.testing.assert_close(before.predictions[0].baseline, after.predictions[0].baseline, rtol=0, atol=0)
            self.assertFalse(torch.equal(before.predictions[0].target, after.predictions[0].target))
            record = {"audio": torch.arange(17).float() + 1, "sample_rate": 8000}
            before = waveform_question(source.reader, record, 0)
            record["audio"][before.inputs[0][1][0] == 0] += 10
            after = waveform_question(source.reader, record, 0)
            torch.testing.assert_close(before.inputs[0][1][0], after.inputs[0][1][0], rtol=0, atol=0)
            self.assertFalse(torch.equal(before.predictions[0].target, after.predictions[0].target))

    def test_sensor_future_and_native_future_are_excluded(self):
        record = {"sensor": torch.randn(128, 9)}
        before = sensor_question(record, 0)
        record["sensor"][96:] += 20
        after = sensor_question(record, 0)
        torch.testing.assert_close(before.inputs[0][1][0], after.inputs[0][1][0], rtol=0, atol=0)
        reader = object.__new__(NativeSource)
        reader.device, reader.dtype, reader.omitted_inputs = torch.device("cpu"), torch.float32, set()
        reader.tokenizer = make_tokenizer()
        observations = [MultimodalObservation(text="red", feedback=torch.tensor([0., 0., 0.]))]
        observations += [MultimodalObservation(previous_action=a, feedback=torch.tensor([1., 0., 0.])) for a in (1, 3, 2)]
        episode = MultimodalEpisode("episode", "world", observations, [1, 3, 2])
        first = history_question(reader, (episode, 2), "first_action", 0)
        last = history_question(reader, (episode, 2), "last_action", 0)
        self.assertEqual((first.answer, last.answer), ("1", "3"))
        episode.actions[2] = 4
        observations[3].feedback.fill_(-100)
        following = history_question(reader, (episode, 2), "last_action", 0)
        self.assertEqual(last.answer, following.answer)
        for (_, old), (_, new) in zip(last.inputs, following.inputs):
            for a, b in zip(old, new):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shogi_candidate_truth_and_complete_successor(self):
        from types import SimpleNamespace
        reader = SimpleNamespace(device=torch.device("cpu"))
        record = SimpleNamespace(position_sfen=shogi.Board().sfen())
        for seed in (8, 9):
            legal = shogi_question(reader, record, "legal", seed, 0)
            illegal = shogi_question(reader, record, "illegal", seed, 0)
            move = legal.prompt.split()[2]
            truth = shogi.Board().is_legal(shogi.Move.from_usi(move))
            self.assertEqual(legal.answer, "yes" if truth else "no")
            self.assertNotEqual(legal.answer, illegal.answer)
        after = shogi_question(reader, record, "after_move", 8, 0)
        move = after.prompt.removesuffix(".").split()[-1]
        board = shogi.Board()
        board.push_usi(move)
        for prediction, expected in zip(after.predictions, board_labels(board, "cpu")):
            torch.testing.assert_close(prediction.target, expected, rtol=0, atol=0)

    def test_all_forms_resume_and_evaluation_preserves_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = make_model()
            source = build_sources(model, make_tokenizer(), question_recipe(root), root)["mnist"]
            for _ in range(5):
                source.loss()
            state = copy.deepcopy(source.state_dict())
            expected = source.loss().detach()
            source.load_state_dict(state)
            panel = make_panel({"mnist": source}, 4)
            for row in panel["mnist"]:
                row["generate"] = False
            rng = torch.get_rng_state().clone()
            result = evaluate_panel(model, {"mnist": source}, panel)
            self.assertEqual(source.step, 5)
            self.assertEqual(source.counts, state["counts"])
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            torch.testing.assert_close(source.loss().detach(), expected, rtol=0, atol=0)
            self.assertIn("sum/1", result["mnist"]["forms"])

    def test_observation_omission_removes_data_without_changing_the_question_or_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sources(make_model(), make_tokenizer(), question_recipe(root), root)["mnist"]
            case = next(row for row in make_panel({"mnist": source}, 2)["mnist"] if row["form"] == "same")
            observations = []
            handle = source.model.input_heads["rgb"].register_forward_hook(lambda *args: observations.append(True))
            set_case(source, {**case, "generate": False})
            source.loss()
            complete = source.last_response["responses"][0]
            self.assertEqual(len(observations), 2)
            observations.clear()
            set_case(source, {**case, "generate": False, "omit_observations": True})
            source.loss()
            missing = source.last_response["responses"][0]
            handle.remove()
            self.assertEqual(observations, [])
            self.assertEqual((complete["prompt"], complete["expected"]), (missing["prompt"], missing["expected"]))
            self.assertGreater(complete["prefix_tokens"], missing["prefix_tokens"])

    def test_each_added_comparison_form_sees_both_pair_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sources(make_model(), make_tokenizer(), question_recipe(root), root)["mnist"]
            seen = {form: set() for form in ("same", "different")}
            for _ in range(4 * (len(source.forms) - 1)):
                source.loss()
                if source.last_form in seen:
                    seen[source.last_form].add(source.last_response["responses"][0]["expected"])
            self.assertEqual(seen, {"same": {"yes", "no"}, "different": {"yes", "no"}})

    def test_both_conditions_start_identically_and_read_the_same_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models, sources = [], []
            for mode in ("fixed", "varied"):
                model = make_model()
                sources.append(build_sources(model, make_tokenizer(), question_recipe(root, mode), root)["mnist"])
                models.append(model)
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for _ in range(8):
                for source in sources:
                    source.loss()
                self.assertEqual(sources[0].last_response["record_indices"], sources[1].last_response["record_indices"])

    def test_new_record_sources_use_waveforms_signals_and_complete_passages(self):
        import intrep.problems.shared_prediction.record_sources  # noqa: F401
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root)
            entries = []
            for split, subject in (("train", 1), ("validation", 2)):
                for label in range(2):
                    path = root / f"{split}-{label}.wav"
                    with wave.open(str(path), "wb") as handle:
                        handle.setnchannels(1)
                        handle.setsampwidth(2)
                        handle.setframerate(8000)
                        handle.writeframes((np.arange(32, dtype=np.int16) * (label + 1)).tobytes())
                    entries.append({"path": path.name, "label": label, "speaker": split, "split": split,
                                    "sample_rate": 8000, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                np.savez(root / f"{split}.npz", signals=np.ones((6, 128, 9), dtype=np.float32),
                         labels=np.arange(6), subjects=np.full(6, subject))
                (root / f"{split}.jsonl").write_text(json.dumps({"passage": "one two three four " * 40, "question": "one two", "answer": True}) + "\n")
            (root / "manifest.json").write_text(json.dumps({"records": entries}))
            (root / "normalization.json").write_text(json.dumps({"mean": [0] * 9, "std": [1] * 9}))
            recipe["sources"].extend([
                {"name": "spoken_digits", "kind": "spoken_digits", "manifest": "manifest.json", "split": "train", "evaluation": {"split": "validation"}},
                {"name": "inertial_activity", "kind": "inertial_activity", "path": "train.npz", "normalization": "normalization.json", "evaluation": {"path": "validation.npz"}},
                {"name": "boolq", "kind": "boolq", "path": "train.jsonl", "records_per_update": 1, "evaluation": {"path": "validation.jsonl"}},
            ])
            model = make_model()
            sources = build_sources(model, make_tokenizer(), recipe, root)
            panel = make_panel({name: sources[name] for name in ("spoken_digits", "inertial_activity", "boolq")}, 1)
            for name, cases in panel.items():
                source = sources[name]
                for case in cases:
                    if case["wording"]:
                        continue
                    set_case(source, {**case, "generate": False})
                    model.zero_grad(set_to_none=True)
                    loss = source.loss()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertIsNotNone(model.core.body.layers[0].feed_forward.w2.weight.grad)
                    if name == "boolq":
                        self.assertIn("one two three four " * 40, source.last_response["responses"][0]["prompt"])
            broken = root / entries[0]["path"]
            broken.write_bytes(broken.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                sources["spoken_digits"].reader.read_record(0)

    def test_question_checkpoint_exact_resume_and_measured_budget(self):
        from transformers import Lfm2ForCausalLM

        from intrep.problems.shared_prediction.training import load_checkpoint, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root)
            Lfm2ForCausalLM(make_model().core.body.config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            common = {"recipe": recipe, "root": root, "optimizer": "adamw", "prompts": [], "evaluation_examples": 1}
            checkpoint = train(base=root / "base", output=root / "first", steps=10, training_seconds=1e-9, **common)
            initial = json.loads((root / "first/result.json").read_text())
            self.assertEqual(initial["completed_steps"], 1)
            resumed = train(base=None, resume=checkpoint, output=root / "resumed", steps=3, **common)
            straight = train(base=root / "base", output=root / "straight", steps=3, **common)
            a, _, state = load_checkpoint(resumed)
            b, _, _ = load_checkpoint(straight)
            for actual, expected in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(state["sources"]["mnist"]["step"], 1)


if __name__ == "__main__":
    unittest.main()
