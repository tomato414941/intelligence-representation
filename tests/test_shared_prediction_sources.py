from __future__ import annotations

import copy
import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from intrep.experience.multimodal.records import (
    MultimodalEpisode,
    save_episode,
    write_selection,
)
from intrep.problems.shared_prediction.recipe import validate_extension, validate_recipe
from intrep.problems.shared_prediction.sources import NativeSource, build_sources
from intrep.problems.shared_prediction.streams import EpochSampler, LineStream
from intrep.representation.inputs.multimodal_observation import MultimodalObservation


def make_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "one", "two", "three", "four", "go", "left", "right", "red"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                        eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for message in messages %}[BOS] {{ message['content'] }} [EOS] {% endfor %}"
    return tokenizer


def make_model():
    from transformers import Lfm2Config, Lfm2ForCausalLM

    from intrep.representation.cores.lfm import split_lfm
    torch.manual_seed(18)
    config = Lfm2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                        num_attention_heads=2, num_key_value_heads=1, layer_types=["conv", "full_attention", "conv"],
                        block_auto_adjust_ff_dim=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        tie_word_embeddings=True, attn_implementation="eager")
    return split_lfm(Lfm2ForCausalLM(config))


def make_recipe(root: Path):
    for split in ("train", "eval"):
        (root / f"{split}.txt").write_text("one two three four\ngo left red right\n")
        images = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
        with gzip.open(root / f"{split}-images.gz", "wb") as handle:
            handle.write(struct.pack(">IIII", 2051, 4, 4, 4) + images.tobytes())
        with gzip.open(root / f"{split}-labels.gz", "wb") as handle:
            handle.write(struct.pack(">II", 2049, 4) + bytes([0, 1, 2, 3]))
    return {"seed": 47, "defaults": {"patch_size": 2, "audio_chunk_size": 4, "action_count": 5,
                                      "block_tokens": 4, "seed": 47},
            "sources": [{"name": "text_data", "kind": "text", "path": "train.txt", "evaluation": {"path": "eval.txt"}},
                        {"name": "pictures", "kind": "idx", "images": "train-images.gz", "labels": "train-labels.gz",
                         "evaluation": {"images": "eval-images.gz", "labels": "eval-labels.gz"}}]}


class StreamTests(unittest.TestCase):
    def test_full_conversations_cover_every_usable_message_without_selection_caps(self):
        from scripts.prepare_joint_conversations import conversation_branches
        def message(identifier, parent, role, **kwargs):
            return {"message_id": identifier, "parent_id": parent, "role": role,
                    "message_tree_id": "root", "text": "long " * 400, "lang": "de", "rank": 8, **kwargs}
        messages = {row["message_id"]: row for row in [
            message("root", None, "prompter"), message("a", "root", "assistant"),
            message("b", "root", "assistant"), message("c", "a", "prompter"),
            message("deleted", "c", "assistant", deleted=True),
            message("orphaned", "deleted", "prompter"),
        ]}
        branches, usable = conversation_branches(messages)
        self.assertEqual(usable, 4)
        self.assertEqual([[row["message_id"] for row in chain] for chain in branches],
                         [["root", "b"], ["root", "a", "c"]])

    def test_complete_population_before_repeat_and_exact_sampler_resume(self):
        sampler = EpochSampler(7, 9)
        seen = [sampler.next() for _ in range(3)]
        state = sampler.state_dict()
        restored = EpochSampler(7, 99)
        restored.load_state_dict(state)
        self.assertEqual([sampler.next() for _ in range(4)], [restored.next() for _ in range(4)])
        self.assertEqual(len(set(seen + state["order"][3:].tolist())), 7)
        self.assertEqual([sampler.next() for _ in range(8)], [restored.next() for _ in range(8)])

    def test_file_ranges_never_read_evaluation_and_resume_at_utf8_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            train = "一つ\n二つ\n".encode()
            path.write_bytes(train + "評価\n".encode())
            source = LineStream(path, end=len(train))
            self.assertEqual(source.next().strip(), "一つ")
            state = source.state_dict()
            self.assertEqual(source.next().strip(), "二つ")
            source.load_state_dict(state)
            self.assertEqual(source.next().strip(), "二つ")
            self.assertEqual(source.next().strip(), "一つ")
            self.assertEqual(source.epochs, 1)
            with self.assertRaisesRegex(ValueError, "complete lines"):
                LineStream(path, start=1)

    def test_extension_cannot_silently_drop_an_existing_dataset(self):
        recipe = {"sources": [{"name": "first", "kind": "text"}, {"name": "second", "kind": "text"}]}
        with self.assertRaisesRegex(ValueError, "retain every"):
            validate_extension(recipe, {"sources": [recipe["sources"][0]]})
        validate_extension(recipe, {"sources": [*recipe["sources"], {"name": "new", "kind": "text"}]})


class SourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import transformers  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("install the lfm extra for source integration tests") from error
        torch.set_num_threads(1)

    def test_sources_keep_all_examples_and_reproduce_the_next_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            validate_recipe(recipe, root)
            model = make_model()
            sources = build_sources(model, make_tokenizer(), recipe, root)
            self.assertEqual(sources["pictures"].provenance()["examples"], 4)
            for source in sources.values():
                state = copy.deepcopy(source.state_dict())
                first = source.loss().detach()
                source.load_state_dict(state)
                second = source.loss().detach()
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            from intrep.problems.shared_prediction.training import evaluate
            original_loss = sources["pictures"].loss
            def augmented_loss():
                torch.rand(5)
                return original_loss()
            sources["pictures"].loss = augmented_loss
            rng = torch.get_rng_state().clone()
            evaluate(model, {"pictures": sources["pictures"]})
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            invalid = copy.deepcopy(recipe)
            invalid["sources"][0]["evaluation"]["path"] = "train.txt"
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_recipe(invalid, root)

    def test_native_forecast_logits_never_observe_future_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            def episode(name, world):
                observations = [MultimodalObservation(text="go left", image=torch.rand(4, 4, 3), audio=torch.rand(8))]
                for action in (0, 1):
                    observations.append(MultimodalObservation(text="red", image=torch.rand(4, 4, 3), audio=torch.rand(8),
                                                              previous_action=action, feedback=torch.tensor([0.1, 0, 0])))
                return MultimodalEpisode(name, world, observations, [0, 1], [0, 1], ["left", "right"])
            train_path = save_episode(root / "episodes", episode("train", "train_world"))
            eval_path = save_episode(root / "episodes", episode("eval", "eval_world"))
            selection = write_selection(root, {"train": [train_path], "validation": [eval_path]})
            recipe["sources"].append({"name": "experience", "kind": "native", "selection": selection.name,
                                       "evaluation": {"split": "validation"}})
            model = make_model()
            sources = build_sources(model, make_tokenizer(), recipe, root)
            source: NativeSource = sources["experience"]
            captured = []
            handle = model.output_heads["next_image"].register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
            first_loss = source.loss()
            first_loss.backward()
            self.assertTrue(torch.isfinite(first_loss))
            for layer in model.core.body.layers:
                self.assertGreater(float(layer.feed_forward.w2.weight.grad.abs().sum()), 0)
            # Change a label after loading the same episode, keeping observations fixed.
            source.transition = 0
            source.episode.observations[1].image.fill_(0)
            second_loss = source.loss()
            handle.remove()
            torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
            self.assertNotEqual(float(first_loss.detach()), float(second_loss.detach()))

    def test_complete_checkpoint_resumes_all_sources_and_can_add_a_new_head(self):
        from transformers import Lfm2ForCausalLM

        from intrep.problems.shared_prediction.training import load_checkpoint, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            # Save an ordinary local HF checkpoint, before separating its heads.
            config = make_model().core.body.config
            torch.manual_seed(31)
            Lfm2ForCausalLM(config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            checkpoint = train(base=root / "base", recipe=recipe, root=root, output=root / "first", steps=1)
            resumed = train(base=None, resume=checkpoint, recipe=recipe, root=root, output=root / "resumed", steps=2)
            straight = train(base=root / "base", recipe=recipe, root=root, output=root / "straight", steps=2)
            resumed_model, _, state = load_checkpoint(resumed)
            straight_model, _, _ = load_checkpoint(straight)
            for actual, expected in zip(resumed_model.parameters(), straight_model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(state["trainer"]["steps"], 2)
            extended = copy.deepcopy(recipe)
            extra = {**copy.deepcopy(recipe["sources"][1]), "name": "another_output"}
            extended["sources"].append(extra)
            extensions = ("tests.test_shared_prediction_sources",)
            path = train(base=None, resume=resumed, recipe=extended, root=root, output=root / "extended", steps=3,
                         extend=True, extensions=extensions)
            with self.assertRaisesRegex(ValueError, "extensions differ"):
                load_checkpoint(path)
            model, _, state = load_checkpoint(path, extensions=extensions)
            self.assertIn("another_output", model.output_heads)
            self.assertEqual(set(state["sources"]), {"text_data", "pictures", "another_output"})
            self.assertEqual(state["sources"]["pictures"]["samples"], 3)
            self.assertEqual(state["sources"]["another_output"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
