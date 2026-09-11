from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from intrep.problems.shared_prediction.evaluation import evaluate_panel, make_panel
from intrep.problems.shared_prediction.sources import build_sources
from tests.test_shared_prediction_sources import make_model, make_tokenizer


def assistant_tokenizer():
    tokenizer = make_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}[BOS] "
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }} [EOS]{% endgeneration %}"
        "{% else %}{{ message['content'] }} [EOS]{% endif %} {% endfor %}"
    )
    return tokenizer


def conversation_recipe(root, messages, **config):
    row = {"id": "branch", "group_id": "tree", "messages": messages}
    for split in ("train", "validation"):
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    return {"seed": 47, "defaults": {"seed": 47, "block_tokens": 4}, "sources": [{
        "name": "conversations", "kind": "conversations", "path": "train.jsonl",
        "evaluation": {"path": "validation.jsonl"}, "conversation_objective": "assistant", **config,
    }]}


class AssistantConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import transformers  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("install the lfm extra") from error
        torch.set_num_threads(1)

    def test_only_assistant_content_and_end_tokens_are_targets(self):
        messages = [{"role": role, "content": text} for role, text in (
            ("system", "go"), ("user", "one two"), ("assistant", "red"),
            ("user", "three four"), ("assistant", "left right"),
        )]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, tokenizer = make_model(), assistant_tokenizer()
            source = build_sources(model, tokenizer, conversation_recipe(root, messages), root)["conversations"]
            record = source.next_record()
            targets = record["tokens"][record["mask"]]
            self.assertEqual(tokenizer.decode(targets), "red [EOS] left right [EOS]")
            self.assertEqual(record["start"], 0)
            self.assertEqual(record["end"], record["tokens"].numel())
            actual = source.record_loss(record)
            hidden = model(model.encode("text", record["tokens"][:, :-1]))
            logits = model.decode("text", hidden)
            expected = F.cross_entropy(logits[record["mask"][:, 1:]], record["tokens"][:, 1:][record["mask"][:, 1:]])
            torch.testing.assert_close(actual, expected)
            actual.backward()
            # Context tokens receive causal gradients even though they are not labels.
            embedding = model.input_heads["text"]
            self.assertGreater(float(embedding.weight.grad[tokenizer.convert_tokens_to_ids("one")].norm()), 0)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_overlapping_windows_cover_every_answer_token_once_and_resume_exactly(self):
        messages = [{"role": "user", "content": "one " * 17},
                    {"role": "assistant", "content": "red " * 37},
                    {"role": "user", "content": "two " * 19},
                    {"role": "assistant", "content": "left " * 23}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = assistant_tokenizer()
            recipe = conversation_recipe(root, messages, conversation_tokens=13, conversation_overlap=5)
            source = build_sources(make_model(), tokenizer, recipe, root)["conversations"]
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                      return_assistant_tokens_mask=True)
            wanted = [index for index, value in enumerate(rendered["assistant_masks"]) if value and index > 0]
            observed = []
            while len(observed) < len(wanted):
                state = copy.deepcopy(source.state_dict())
                record = source.next_record()
                source.load_state_dict(state)
                repeat = source.next_record()
                torch.testing.assert_close(record["tokens"], repeat["tokens"], rtol=0, atol=0)
                torch.testing.assert_close(record["mask"], repeat["mask"], rtol=0, atol=0)
                self.assertLessEqual(record["tokens"].numel(), 13)
                observed.extend(record["start"] + index for index, value in enumerate(record["mask"][0]) if value)
            self.assertEqual(observed, wanted)
            self.assertEqual(len(observed), len(set(observed)))

    def test_question_forms_and_evaluation_restore_conversation_cursor(self):
        messages = [{"role": "user", "content": "one two three four"},
                    {"role": "assistant", "content": "go left red right"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sources(make_model(), assistant_tokenizer(),
                                   conversation_recipe(root, messages, question_mode="varied"), root)["conversations"]
            for _ in range(5):
                source.loss()
            state = copy.deepcopy(source.state_dict())
            expected = source.loss().detach()
            source.load_state_dict(state)
            panel = make_panel({"conversations": source}, 2)
            measured = evaluate_panel(source.model, {"conversations": source}, panel, generate_answers=False)
            self.assertEqual(source.state_dict(), state)
            torch.testing.assert_close(source.loss().detach(), expected, rtol=0, atol=0)
            self.assertIn("first_word/1", measured["conversations"]["forms"])

    def test_rejects_missing_generation_mask_and_all_unanswered_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = conversation_recipe(root, [{"role": "user", "content": "one two"}])
            with self.assertRaisesRegex(ValueError, "generation spans"):
                build_sources(make_model(), make_tokenizer(), recipe, root)
            source = build_sources(make_model(), assistant_tokenizer(), recipe, root)["conversations"]
            with self.assertRaisesRegex(ValueError, "no assistant targets"):
                source.loss()


if __name__ == "__main__":
    unittest.main()
