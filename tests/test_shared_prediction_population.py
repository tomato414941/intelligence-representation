from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import shogi
import torch

from intrep.experience.multimodal.records import MultimodalEpisode, save_episode, write_selection
from intrep.problems.shared_prediction.full_evaluation import evaluate_full, paired_full_comparison
from intrep.problems.shared_prediction.population import completed_epochs, limit_epochs, population_records, rewind_population
from intrep.problems.shared_prediction.recipe import evaluation_recipe, without_evaluation_sampling
from intrep.problems.shared_prediction.sources import build_sources
from intrep.problems.shared_prediction.streams import EpochSampler, LineStream
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from tests.test_assistant_conversations import assistant_tokenizer, conversation_recipe
from tests.test_shared_prediction_questions import question_recipe
from tests.test_shared_prediction_sources import make_model, make_recipe, make_tokenizer


class PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_finite_streams_stop_at_boundary_and_can_continue_into_another_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.txt"
            path.write_text("one\n\ntwo\n\n")
            stream = LineStream(path)
            stream.epoch_limit = 1
            self.assertEqual(list(iter(stream.next, None)), ["one\n", "two\n"])
            self.assertEqual((stream.offset, stream.records), (path.stat().st_size, 2))
            stream.epoch_limit = 2
            self.assertEqual(list(iter(stream.next, None)), ["one\n", "two\n"])
            self.assertEqual(stream.epochs, 1)
            sampler = EpochSampler(5, 47)
            sampler.epoch_limit = 1
            self.assertEqual(set(iter(sampler.next, None)), set(range(5)))
            restored = EpochSampler(5, 47)
            restored.load_state_dict(sampler.state_dict())
            sampler.epoch_limit = restored.epoch_limit = 2
            self.assertEqual(list(iter(sampler.next, None)), list(iter(restored.next, None)))

    def test_text_pass_includes_final_short_block_and_resumes_without_losing_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            (root / "train.txt").write_text("one two three four\ngo left red\n\n")
            source = build_sources(make_model(), make_tokenizer(), recipe, root)["text_data"]
            limit_epochs(source, 1)
            first = source.next_record()
            state = copy.deepcopy(source.state_dict())
            self.assertEqual(completed_epochs(source), 0)
            tail = source.next_record()
            self.assertEqual(tail.shape[1] - 1, 2)
            self.assertEqual(completed_epochs(source), 1)
            with self.assertRaises(StopIteration):
                source.next_record()
            source.load_state_dict(state)
            torch.testing.assert_close(source.next_record(), tail, rtol=0, atol=0)
            recovered = torch.cat((first[0], tail[0, 1:])).tolist()
            self.assertEqual(recovered, source.text_ids("one two three four go left red"))
            self.assertEqual(source.progress()["trained_tokens"], 6)
            # The final block can share an update with a longer block.
            source.record_batch_loss([first, tail]).backward()

    def test_full_evaluation_covers_all_image_anchors_and_question_forms(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = question_recipe(root)
            count = 35
            with gzip.open(root / "eval-images.gz", "wb") as handle:
                handle.write(struct.pack(">IIII", 2051, count, 4, 4) + np.arange(count * 16, dtype=np.uint8).tobytes())
            with gzip.open(root / "eval-labels.gz", "wb") as handle:
                handle.write(struct.pack(">II", 2049, count) + bytes(index % 10 for index in range(count)))
            recipe["defaults"].update(question_evaluation_examples=8)
            model = make_model()
            source = build_sources(model, make_tokenizer(), evaluation_recipe(recipe), root)["mnist"]
            source.loss()
            state = copy.deepcopy(source.state_dict())
            rng = torch.get_rng_state().clone()
            first = evaluate_full(model, {"mnist": source}, root / "before", generate_answers=False)
            self.assertEqual(source.step, state["step"])
            torch.testing.assert_close(source.sampler.order, state["reader"]["order"], rtol=0, atol=0)
            self.assertEqual(source.sampler.cursor, state["reader"]["cursor"])
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            second = evaluate_full(model, {"mnist": source}, root / "after", generate_answers=False)
            self.assertEqual(first, second)
            self.assertEqual(first["mnist"]["records"], count)
            self.assertEqual(first["mnist"]["examples"], count * 13)
            rows = [json.loads(line) for line in (root / "before" / first["mnist"]["rows_file"]).read_text().splitlines()]
            for row in rows:
                if row["form"] in ("original", "inpaint"):
                    self.assertEqual(row["response"]["record_indices"], [int(row["record_key"].split(":")[1])])
            for form in source.forms:
                for wording in (0,) if form == "original" else (0, 1):
                    selected = [row for row in rows if (row["form"], row["wording"]) == (form, wording)]
                    self.assertEqual({row["record_key"] for row in selected}, {f"record:{index}" for index in range(count)})
            paired = paired_full_comparison(first, second, before_directory=root / "before", after_directory=root / "after")
            self.assertEqual(paired["mnist"]["paired_change"]["loss"], {"mean": 0.0, "count": count * 13})
            # Incomplete result files must never pass as a full paired evaluation.
            path = root / "after" / second["mnist"]["rows_file"]
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "row counts"):
                paired_full_comparison(first, second, before_directory=root / "before", after_directory=root / "after")

    def test_full_conversation_evaluation_scores_every_assistant_target_exactly_once(self):
        messages = [{"role": "user", "content": "one " * 19},
                    {"role": "assistant", "content": "red " * 41},
                    {"role": "user", "content": "two " * 7},
                    {"role": "assistant", "content": "left " * 31}]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            tokenizer = assistant_tokenizer()
            recipe = conversation_recipe(root, messages, conversation_tokens=13, conversation_overlap=5)
            model = make_model()
            source = build_sources(model, tokenizer, evaluation_recipe(recipe), root)["conversations"]
            state = copy.deepcopy(source.state_dict())
            measured = evaluate_full(model, {"conversations": source}, root / "evaluation", generate_answers=False)
            rows = [json.loads(line) for line in (root / "evaluation" / measured["conversations"]["rows_file"]).read_text().splitlines()]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
            expected = sum(encoded["assistant_masks"][1:])
            self.assertEqual(sum(row["response"]["target_tokens"] for row in rows), expected)
            self.assertGreater(len(rows), 1)
            self.assertEqual(rows[-1]["window"][1], len(encoded["input_ids"]))
            self.assertEqual(source.state_dict(), state)

    def test_partial_question_batches_do_not_wrap_or_drop_the_last_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root, mode="fixed")
            recipe["sources"][0]["records_per_update"] = 4
            recipe["sources"][1]["records_per_update"] = 6
            sources = build_sources(make_model(), make_tokenizer(), recipe, root)
            for source in sources.values():
                limit_epochs(source, 1)
            sources["text_data"].loss().backward()
            self.assertEqual(sources["text_data"].last_update_info["records"], 2)
            sources["mnist"].loss().backward()
            self.assertEqual(sources["mnist"].sampler.samples, 3)
            sources["mnist"].loss().backward()
            self.assertEqual(sources["mnist"].sampler.samples, 4)
            self.assertEqual(sources["mnist"].last_update_info["records"], 2)
            self.assertTrue(all(completed_epochs(source) == 1 for source in sources.values()))
            for source in sources.values():
                with self.assertRaises(StopIteration):
                    source.loss()

    def test_full_pass_covers_every_episode_transition_and_all_other_record_sources(self):
        import intrep.problems.shared_prediction.record_sources  # noqa: F401
        from intrep.problems.shared_prediction.replay import batch_loss, pack_batch, unpack_batch
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = question_recipe(root)
            episodes = []
            for number in range(2):
                actions = [0, 1, 2]
                observations = [MultimodalObservation(text="one", image=torch.rand(4, 4, 3), audio=torch.rand(8))]
                observations += [MultimodalObservation(text="red", image=torch.rand(4, 4, 3), audio=torch.rand(8),
                                                      previous_action=action, feedback=torch.tensor([0.1, 0, 0]))
                                 for action in actions]
                episode = MultimodalEpisode(f"episode-{number}", f"world-{number}", observations, actions)
                episodes.append(save_episode(root / "episodes", episode))
            selection = write_selection(root, {"validation": episodes})
            entries = []
            for label in range(2):
                path = root / f"{label}.wav"
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes((np.arange(32, dtype=np.int16) * (label + 1)).tobytes())
                entries.append({"path": path.name, "label": label, "speaker": "validation", "split": "validation",
                                "sample_rate": 8000, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (root / "manifest.json").write_text(json.dumps({"records": entries}))
            np.savez(root / "signals.npz", signals=np.ones((6, 128, 9), dtype=np.float32), labels=np.arange(6), subjects=np.full(6, 2))
            (root / "normalization.json").write_text(json.dumps({"mean": [0] * 9, "std": [1] * 9}))
            (root / "boolq.jsonl").write_text("".join(json.dumps({"passage": "one two three four " * 40,
                "question": question, "answer": True}) + "\n" for question in ("one", "two", "three")))
            board = shogi.Board()
            moves = [move.usi() for move in board.legal_moves]
            (root / "shogi.jsonl").write_text("".join(json.dumps({"position_sfen": board.sfen(), "legal_moves": moves,
                "chosen_move": move, "game_index": index}) + "\n" for index, move in enumerate(moves[:2])))
            recipe["sources"] = [
                {"name": "experience", "kind": "native", "selection": selection.name, "split": "validation",
                 "records_per_update": 4},
                {"name": "spoken_digits", "kind": "spoken_digits", "manifest": "manifest.json", "split": "validation"},
                {"name": "inertial_activity", "kind": "inertial_activity", "path": "signals.npz", "normalization": "normalization.json"},
                {"name": "boolq", "kind": "boolq", "path": "boolq.jsonl", "records_per_update": 1},
                {"name": "shogi", "kind": "shogi_examples", "path": "shogi.jsonl"},
            ]
            recipe["defaults"].update(question_evaluation_examples=1, question_evaluation_worlds=1)
            model = make_model()
            sources = build_sources(model, make_tokenizer(), recipe, root)
            measured = evaluate_full(model, sources, root / "evaluation", generate_answers=False)
            self.assertEqual({name: value["records"] for name, value in measured.items()},
                             {"experience": 6, "spoken_digits": 2, "inertial_activity": 6, "boolq": 3, "shogi": 2})
            self.assertEqual(measured["experience"]["groups"], 2)
            self.assertEqual(measured["experience"]["examples"], 6 * 7)
            for source in sources.values():
                rewind_population(source)
                rows = list(population_records(source))
                self.assertEqual(completed_epochs(source), 1)
                self.assertEqual(len(rows), measured[source.config["name"]]["records"])
                # Replay media from disk and structured targets from a safe
                # checkpoint, without consuming the remaining fresh records.
                rewind_population(source)
                batch = source.next_batch()
                expected = batch_loss(source, batch).detach()
                cursor = copy.deepcopy(source.reader.progress())
                buffer = io.BytesIO()
                torch.save(pack_batch(source, batch), buffer)
                buffer.seek(0)
                restored = unpack_batch(source, torch.load(buffer, weights_only=True))
                torch.testing.assert_close(batch_loss(source, restored).detach(), expected, rtol=0, atol=0)
                self.assertEqual(source.reader.progress(), cursor)

    def test_default_training_finishes_all_sources_and_partial_epoch_resumes_exactly(self):
        from transformers import Lfm2ForCausalLM
        from intrep.problems.shared_prediction.training import load_checkpoint, train
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            (root / "train.txt").write_text("one two three four\ngo left red\n")
            config = make_model().core.body.config
            Lfm2ForCausalLM(config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            options = {"root": root, "optimizer": "adamw", "prompts": []}
            recipe["defaults"].update(question_evaluation_examples=8, question_evaluation_worlds=4)
            partial = train(base=root / "base", recipe=recipe, output=root / "partial", training_seconds=1e-9, **options)
            partial_report = json.loads((root / "partial/result.json").read_text())
            self.assertFalse(partial_report["population_complete"])
            uncapped = without_evaluation_sampling(recipe)
            resumed = train(base=None, resume=partial, recipe=uncapped, output=root / "resumed", **options)
            straight = train(base=root / "base", recipe=uncapped, output=root / "straight", **options)
            resumed_model, _, resumed_state = load_checkpoint(resumed)
            straight_model, _, straight_state = load_checkpoint(straight)
            for actual, expected in zip(resumed_model.parameters(), straight_model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(resumed_state["trainer"]["steps"], straight_state["trainer"]["steps"])
            report = json.loads((root / "straight/result.json").read_text())
            self.assertTrue(report["population_complete"])
            self.assertEqual(report["requested_epochs"], 1)
            self.assertIsNone(report["requested_steps"])
            self.assertTrue(all(value >= 1 for value in report["completed_epochs"].values()))
            self.assertEqual(report["completed_epochs"]["pictures"], 1)
            self.assertGreaterEqual(report["source_progress"]["text_data"]["trained_tokens"], 6)
            self.assertEqual(report["source_progress"]["pictures"]["samples"], 4)
            steps = [json.loads(line) for line in (root / "straight/steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["source"] for row in steps if row["experience"] == "fresh"],
                             ["text_data", "pictures", "text_data", "pictures", "pictures", "pictures"])
            self.assertEqual(sum(row["experience"] == "replay" for row in steps), 2)
            self.assertEqual(report["source_progress"]["text_data"]["trained_tokens"], 6)
            self.assertIsNone(report["joint_updates"])
            self.assertEqual(report["paired_evaluation"]["pictures"]["examples"], 4)


if __name__ == "__main__":
    unittest.main()
