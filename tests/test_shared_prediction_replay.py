from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.population import completed_epochs
from intrep.problems.shared_prediction.replay import ExperienceReplay, batch_loss, pack_batch, unpack_batch
from intrep.problems.shared_prediction.sources import build_sources
from tests.test_assistant_conversations import assistant_tokenizer, conversation_recipe
from tests.test_shared_prediction_questions import question_recipe
from tests.test_shared_prediction_sources import make_model, make_recipe, make_tokenizer


def assert_state_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_state_equal(test, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for value, wanted in zip(actual, expected):
            assert_state_equal(test, value, wanted)
    else:
        test.assertEqual(actual, expected)


class ExperienceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_first_pass_and_fixed_replay_budget_survive_reservoir_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            (root / "train.txt").write_text("one two three four " * 8 + "\n")
            model = make_model()
            sources = build_sources(model, make_tokenizer(), recipe, root)
            replay = ExperienceReplay(sources, capacity=1, every=3)
            trainer = JointTrainer(model, {name: 1 for name in sources}, learning_rate=.001)
            pictures, tokens, kinds = [], [], []
            while not replay.complete:
                name, batch, is_replay = replay.next()
                before = copy.deepcopy(sources[name].state_dict())
                trainer.step({name: lambda: batch_loss(sources[name], batch)})
                if is_replay:
                    assert_state_equal(self, sources[name].state_dict(), before)
                elif name == "pictures":
                    pictures.append(batch["records"][0]["index"])
                else:
                    tokens.extend(batch["records"][0][0, 1:].tolist())
                replay.record_update(name, batch, is_replay)
                kinds.append(is_replay)
            self.assertEqual(sorted(pictures), list(range(4)))
            self.assertEqual(tokens, sources["text_data"].text_ids((root / "train.txt").read_text())[1:])
            self.assertEqual(kinds, [False, False, False, True] * 4)
            self.assertEqual(replay.fresh_updates, {"text_data": 8, "pictures": 4})
            self.assertEqual(sum(replay.replay_updates.values()), 4)
            self.assertEqual(replay.progress()["retained_batches"], {"text_data": 1, "pictures": 1})
            self.assertEqual({name: completed_epochs(source) for name, source in sources.items()},
                             {"text_data": 1, "pictures": 1})
            self.assertEqual(sources["pictures"].sampler.samples, 4)
            self.assertEqual(sources["text_data"].tokens, 31)
            with self.assertRaises(StopIteration):
                replay.next()

    def test_zero_replay_budget_processes_each_requested_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = build_sources(make_model(), make_tokenizer(), make_recipe(root), root)
            replay = ExperienceReplay(sources, every=0, epochs=2)
            while not replay.complete:
                name, batch, is_replay = replay.next()
                self.assertFalse(is_replay)
                batch_loss(sources[name], batch).backward()
                replay.record_update(name, batch, is_replay)
            self.assertEqual(sources["pictures"].sampler.samples, 8)
            self.assertEqual([completed_epochs(source) for source in sources.values()], [2, 2])
            self.assertEqual(sum(replay.replay_updates.values()), 0)
            self.assertTrue(all(not rows for rows in replay.memory.values()))

    def test_replay_preserves_question_forms_pairs_targets_and_fresh_cursors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = build_sources(make_model(), make_tokenizer(), question_recipe(root), root)
            for source in sources.values():
                batches = []
                for _ in range(2 * len(source.forms)):
                    batch = source.next_batch()
                    expected = batch_loss(source, batch).detach()
                    batches.append((pack_batch(source, batch), expected))
                reader_state = copy.deepcopy(source.reader.state_dict())
                partner_states = {key: copy.deepcopy(value.state_dict()) for key, value in source.partners.items()}
                self.assertEqual({batch["form"] for batch, _ in batches}, set(source.forms))
                for packed, expected in batches:
                    # The same safe checkpoint payload reconstructs original and
                    # derived questions, even after their source has advanced.
                    serialized = io.BytesIO()
                    torch.save(packed, serialized)
                    serialized.seek(0)
                    restored = unpack_batch(source, torch.load(serialized, weights_only=True))
                    torch.testing.assert_close(batch_loss(source, restored).detach(), expected, rtol=0, atol=0)
                    assert_state_equal(self, source.reader.state_dict(), reader_state)
                    for key, state in partner_states.items():
                        assert_state_equal(self, source.partners[key].state_dict(), state)

    def test_conversation_replay_keeps_context_windows_and_assistant_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = assistant_tokenizer()
            messages = [{"role": "user", "content": "one " * 17},
                        {"role": "assistant", "content": "red " * 31}]
            recipe = conversation_recipe(root, messages, conversation_tokens=13, conversation_overlap=5)
            source = build_sources(make_model(), tokenizer, recipe, root)["conversations"]
            replay = ExperienceReplay({"conversations": source}, every=1)
            targets = []
            while not replay.complete:
                name, batch, is_replay = replay.next()
                record = batch["records"][0]
                cursor = source.position
                batch_loss(source, batch).backward()
                self.assertEqual(source.position, cursor)
                if not is_replay:
                    targets.extend(record["tokens"][record["mask"]].tolist())
                replay.record_update(name, batch, is_replay)
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                     return_assistant_tokens_mask=True)
            self.assertEqual(targets, [token for index, (token, mask) in enumerate(
                zip(encoded["input_ids"], encoded["assistant_masks"])) if index and mask])
            self.assertGreater(replay.fresh_updates["conversations"], 1)
            for stored in replay.memory["conversations"]:
                for key in ("tokens", "mask"):
                    self.assertEqual(stored["records"][0][key].device.type, "cpu")
                    self.assertFalse(stored["records"][0][key].requires_grad)

    def test_checkpoint_resumes_pending_replay_and_inherits_its_configuration(self):
        from transformers import Lfm2ForCausalLM
        from intrep.problems.shared_prediction.training import load_checkpoint, train

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            (root / "train.txt").write_text("one two three four " * 8 + "\n")
            Lfm2ForCausalLM(make_model().core.body.config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            options = {"recipe": recipe, "root": root, "optimizer": "adamw", "prompts": [], "evaluation_examples": 2}
            partial = train(base=root / "base", output=root / "partial", steps=8,
                            replay_every=2, replay_capacity=1, **options)
            _, _, partial_state = load_checkpoint(partial)
            self.assertEqual(partial_state["experience_replay"]["since_replay"], 2)
            resumed = train(base=None, resume=partial, output=root / "resumed", **options)
            straight = train(base=root / "base", output=root / "straight", replay_every=2, replay_capacity=1, **options)
            a, _, state_a = load_checkpoint(resumed)
            b, _, state_b = load_checkpoint(straight)
            for actual, expected in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in ("trainer", "sources", "experience_replay", "torch_rng"):
                assert_state_equal(self, state_a[key], state_b[key])
            rows = [json.loads(line) for line in (root / "resumed/steps.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["experience"], "replay")
            report = json.loads((root / "resumed/result.json").read_text())
            self.assertTrue(report["population_complete"])
            self.assertEqual(report["experience_replay"]["fresh_updates_per_replay"], 2)
            self.assertEqual(sum(report["experience_replay"]["replay_updates"].values()), 6)
            with self.assertRaisesRegex(ValueError, "same schedule"):
                train(base=None, resume=partial, output=root / "changed", replay_every=3, **options)

    def test_resume_retains_the_requested_number_of_fresh_passes(self):
        from transformers import Lfm2ForCausalLM
        from intrep.problems.shared_prediction.training import load_checkpoint, train

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            Lfm2ForCausalLM(make_model().core.body.config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            options = {"recipe": recipe, "root": root, "prompts": [], "evaluation_examples": 1}
            partial = train(base=root / "base", output=root / "partial", epochs=2, training_seconds=1e-9, **options)
            resumed = train(base=None, resume=partial, output=root / "resumed", **options)
            _, _, state = load_checkpoint(resumed)
            self.assertEqual(state["experience_replay"]["epochs"], 2)
            self.assertEqual(state["sources"]["pictures"]["samples"], 8)
            report = json.loads((root / "resumed/result.json").read_text())
            self.assertEqual(report["completed_epochs"], {"text_data": 2, "pictures": 2})


if __name__ == "__main__":
    unittest.main()
