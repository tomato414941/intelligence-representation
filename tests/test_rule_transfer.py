from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from intrep.problems.shared_prediction.evaluation import make_panel, set_case
from intrep.problems.shared_prediction.rule_transfer import (
    MeasuredReadout,
    compare_counterfactuals,
    digit_question,
    evaluate_transfer,
    order_question,
)
from intrep.problems.shared_prediction.rule_transfer_data import (
    DIGITS,
    PANEL_SCHEMA,
    historical_indices,
    make_orders,
    make_pairs,
    precedes,
    text_training_examples,
    validate_panel,
)
from intrep.problems.shared_prediction.sources import Source, build_sources
from tests.test_shared_prediction_questions import question_recipe
from tests.test_shared_prediction_sources import make_model, make_tokenizer


def fixture_panel():
    labels = np.repeat(np.arange(10), 80)
    images = np.zeros((len(labels), 4, 4), dtype=np.uint8)
    for index, label in enumerate(labels):
        images[index, 0, 0] = label
        images[index, 1, :2] = index % 256, index // 256
    orders = make_orders(47)
    panels = make_pairs(images, labels, orders, per_class_pair=1)
    panel = {"schema_version": PANEL_SCHEMA, "orders": orders, "per_class_pair": 1,
             "excluded_indices": [], "panels": panels,
             "holdout_excluded_indices": [index for row in panels["holdout"] for index in row["indices"]]}
    return panel, images, labels


def digit_tokenizer():
    tokenizer = make_tokenizer()
    tokenizer.add_tokens([*map(str, DIGITS), "yes", "no"])
    return tokenizer


class DataTests(unittest.TestCase):
    def test_balanced_intervention_changes_answers_only_for_the_same_text_examples(self):
        orders = make_orders(47)
        a, b, control = [text_training_examples(orders, condition) for condition in ("a", "b", "control")]
        self.assertEqual(a, text_training_examples(make_orders(47), "a"))
        self.assertEqual([row["digits"] for row in a], [row["digits"] for row in b])
        self.assertEqual([row["digits"] for row in a], [row["digits"] for row in control])
        self.assertEqual(sum(left["answer"] != right["answer"] for left, right in zip(a, b)), 46)
        for examples in (a, b, control):
            self.assertEqual(len(examples), 90)
            self.assertEqual(sum(row["answer"] == "yes" for row in examples), 45)
            self.assertTrue(all("image" not in row for row in examples))
        self.assertTrue(all(row["rule"] == "old" for row in control))
        self.assertTrue(all(row["rule"] == "new" for row in a + b))

    def test_panels_exclude_prior_images_and_exact_duplicates_with_balanced_classes(self):
        panel, images, labels = fixture_panel()
        images[5] = images[0]
        blocked = hashlib.sha256(images[80].tobytes()).hexdigest()
        rows = make_pairs(images, labels, panel["orders"], excluded={0}, blocked_hashes={blocked}, per_class_pair=2)
        indices = [index for cases in rows.values() for row in cases for index in row["indices"]]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(set(indices).isdisjoint({0, 5, 80}))
        for cases in rows.values():
            self.assertEqual(len(cases), 90)
            self.assertEqual(sum(row["changed"] for row in cases), 46)
            counts = np.bincount([int(labels[index]) for row in cases for index in row["indices"]], minlength=10)
            np.testing.assert_array_equal(counts, np.full(10, 18))

    def test_panel_validation_rejects_overlap_changed_images_and_wrong_labels(self):
        panel, images, labels = fixture_panel()
        validate_panel(panel, images, labels)
        broken = copy.deepcopy(panel)
        broken["panels"]["holdout"][0] = copy.deepcopy(broken["panels"]["development"][0])
        with self.assertRaises(ValueError):
            validate_panel(broken, images, labels)
        changed = images.copy()
        index = panel["panels"]["holdout"][0]["indices"][0]
        changed[index, 2, 2] ^= 1
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            validate_panel(panel, changed, labels)
        changed_labels = labels.copy()
        changed_labels[index] = (changed_labels[index] + 1) % 10
        with self.assertRaisesRegex(ValueError, "label changed"):
            validate_panel(panel, images, changed_labels)

    def test_history_includes_both_partners_but_keeps_other_datasets_separate(self):
        history = {"sources": {"mnist": {"rows": [
            {"key": "image:7/original/0", "response": {"record_indices": [7, 29]}}]},
            "fashion_mnist": {"rows": [{"key": "image:88", "response": {"record_indices": [88, 99]}}]}}}
        self.assertEqual(historical_indices(history), {7, 29})

    def test_new_transfer_results_are_also_excluded_when_preparing_a_later_panel(self):
        history = {"schema_version": "intrep.rule_transfer_evaluation.v1", "digit_readouts": [{"index": 7}],
                   "rows": [{"indices": [7, 29]}, {"indices": [29, 7]}]}
        self.assertEqual(historical_indices(history), {7, 29})


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_targets_and_order_table_cannot_change_image_question_inputs(self):
        source = Source(torch.nn.Linear(1, 1), digit_tokenizer(), {}, Path("."))
        orders = make_orders(47)
        a, b = next((a, b) for a in DIGITS for b in DIGITS if precedes(orders["a"], a, b) != precedes(orders["b"], a, b))
        records = [{"label": a, "image": torch.rand(4, 4, 3)}, {"label": b, "image": torch.rand(4, 4, 3)}]
        first = order_question(source, records, order=orders["a"])
        second = order_question(source, records, order=orders["b"])
        self.assertNotEqual(first.answer, second.answer)
        self.assertEqual(first.prompt, second.prompt)
        for (name_a, args_a), (name_b, args_b) in zip(first.inputs, second.inputs):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(args_a[0], args_b[0], rtol=0, atol=0)
        record = {"label": a, "image": records[0]["image"]}
        before = digit_question(record)
        record["label"] = b
        after = digit_question(record)
        torch.testing.assert_close(before.inputs[0][1][0], after.inputs[0][1][0], rtol=0, atol=0)

    def test_real_generation_and_candidate_probabilities_ignore_expected_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = build_sources(make_model(), digit_tokenizer(), question_recipe(Path(directory)), Path(directory))["mnist"].reader
            source.model.eval()
            meter = MeasuredReadout(source, max_tokens=2)
            question = digit_question({"label": 7, "image": torch.rand(4, 4, 3)})
            first = meter(question, tuple(map(str, DIGITS)))
            question.answer = "secret target must never enter the prefix"
            second = meter(question, tuple(map(str, DIGITS)))
            self.assertEqual(first["answer"], second["answer"])
            self.assertEqual(first["probabilities"], second["probabilities"])
            self.assertEqual(first["prefix_tokens"], second["prefix_tokens"])
            self.assertAlmostEqual(sum(first["probabilities"].values()), 1.0, places=6)
            self.assertTrue(0 <= first["candidate_mass"] <= 1)

    def test_reserved_images_are_excluded_from_primary_and_partner_evaluation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = question_recipe(root)
            source = build_sources(make_model(), digit_tokenizer(), recipe, root)["mnist"]
            # Each class has two examples, so a reserved partner still leaves a valid pool.
            source.reader.images = np.repeat(source.reader.images, 2, axis=0)
            source.reader.labels = np.repeat(source.reader.labels, 2)
            from intrep.problems.shared_prediction.questions import QuestionSource
            from intrep.problems.shared_prediction.streams import EpochSampler
            source.reader.sampler = EpochSampler(8, 47)
            source.reader.config["evaluation_excluded_indices"] = [1, 3, 5, 7]
            source = QuestionSource(source.reader)
            panel = make_panel({"mnist": source}, 8)["mnist"]
            self.assertEqual({row["record_key"] for row in panel}, {"image:0", "image:2", "image:4", "image:6"})
            for case in panel:
                set_case(source, case)
                records = source._records(case["seed"])
                self.assertTrue(all(row["index"] % 2 == 0 for row in records))
            source._forced = None
            source.reader.sampler = EpochSampler(8, 47)
            seen = {source._records(47)[0]["index"] for _ in range(8)}
            self.assertEqual(seen, set(range(8)))


class OracleFixtureReadout:
    """Use synthetic pixel codes only to test the evaluator's routing and scoring."""

    def __init__(self, source, order, *, invalid_digit=None):
        self.source, self.order, self.invalid_digit = source, order, invalid_digit
        self.calls = 0
        self.seen_images = set()

    def snapshot(self):
        return {"core_calls": self.calls, "core_input_positions": self.calls, "seconds": self.calls * 0.001}

    def __call__(self, question, candidates):
        assert question.answer is None
        self.calls += 1
        images = [values[0] for name, values in question.inputs if name == "rgb"]
        for image in images:
            self.seen_images.add(round(float(image[1, 0, 0]) * 255) + 256 * round(float(image[1, 1, 0]) * 255))
        if images:
            digits = [round(float(image[0, 0, 0]) * 255) for image in images]
        else:
            digits = [int(self.source.tokenizer.decode(values[0][0].tolist()).split()[-1])
                      for name, values in question.inputs if name == "text"]
        if question.prompt.startswith("Name"):
            target = str(digits[0])
            answer = "invalid" if digits[0] == self.invalid_digit else target
        else:
            order = list(DIGITS) if "old order" in question.prompt else self.order
            target = answer = "yes" if precedes(order, *digits) else "no"
        return {"answer": answer, "probabilities": {candidate: float(candidate == target) for candidate in candidates},
                "candidate_mass": 1.0, "generated_tokens": 1, "prefix_tokens": 1, "seconds": 0.001}


class EvaluatorTests(unittest.TestCase):
    def evaluate(self, panel, images, labels, condition, *, learned_order=None, invalid_digit=None):
        source = Source(torch.nn.Linear(1, 1), digit_tokenizer(), {}, Path("."))
        reader = OracleFixtureReadout(source, learned_order or panel["orders"][condition], invalid_digit=invalid_digit)
        source.model.train()
        result = evaluate_transfer(source, panel, images, labels, split="development", order_name=condition, readout=reader)
        self.assertTrue(source.model.training)
        self.assertTrue(all(parameter.requires_grad for parameter in source.model.parameters()))
        expected = {index for row in panel["panels"]["development"] for index in row["indices"]}
        self.assertEqual(reader.seen_images, expected)
        result["panel_sha256"] = "same-fixture"
        return result

    def test_all_routes_and_all_four_counterfactual_answers_are_scored(self):
        panel, images, labels = fixture_panel()
        a = self.evaluate(panel, images, labels, "a")
        b = self.evaluate(panel, images, labels, "b")
        self.assertTrue(a["transfer_interpretable"])
        self.assertTrue(b["transfer_interpretable"])
        for route in a["summary"].values():
            self.assertEqual(route["accuracy"], 1)
            self.assertEqual(route["both_orientations_correct"], 1)
        compared = compare_counterfactuals(a, b)
        for route in compared.values():
            self.assertEqual(route["changed"], {"pairs": 23, "all_four_correct": 1.0})
            self.assertEqual(route["unchanged"], {"pairs": 22, "all_four_correct": 1.0})
        self.assertEqual(a["costs"]["digit_cache"]["core_calls"], 90)
        self.assertEqual(a["costs"]["text_rule_cache"]["core_calls"], 100)
        self.assertEqual(a["costs"]["direct"]["core_calls"], 90)

    def test_unchanged_rule_and_constant_answers_cannot_pass_the_changed_quartets(self):
        panel, images, labels = fixture_panel()
        a = self.evaluate(panel, images, labels, "a")
        b = self.evaluate(panel, images, labels, "b", learned_order=panel["orders"]["a"])
        compared = compare_counterfactuals(a, b)
        for route in compared.values():
            self.assertEqual(route["changed"]["all_four_correct"], 0)
            self.assertEqual(route["unchanged"]["all_four_correct"], 1)
        for result in (a, b):
            for row in result["rows"]:
                row["answer"] = "yes"
        for route in compare_counterfactuals(a, b).values():
            self.assertEqual(route["changed"]["all_four_correct"], 0)
            self.assertEqual(route["unchanged"]["all_four_correct"], 0)

    def test_invalid_digit_generation_has_no_oracle_fallback_and_fails_the_gate(self):
        panel, images, labels = fixture_panel()
        result = self.evaluate(panel, images, labels, "a", invalid_digit=0)
        self.assertFalse(result["transfer_interpretable"])
        self.assertAlmostEqual(result["gates"]["digit_naming"]["accuracy"], 0.9)
        affected = [row for row in result["rows"] if row["route"] == "read_then_apply" and 0 in row["class_pair"]]
        self.assertTrue(affected)
        self.assertTrue(all(row["answer"] == "" and None in row["read_digits"] for row in affected))
        self.assertEqual(result["summary"]["oracle_digits"]["accuracy"], 1)
        self.assertEqual(result["summary"]["cached_posterior"]["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
