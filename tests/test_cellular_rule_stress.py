import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from intrep.problems.cellular_rule_inference.episodes import (
    replace_context_rules,
    sample_episodes,
    sample_rules,
)
from intrep.problems.cellular_rule_inference.stress import (
    baseline_probabilities,
    corrupt_outputs,
    evaluate_stress,
    probability_metrics,
    splice_context,
)
from intrep.problems.cellular_rule_inference.training import (
    RuleInferenceTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.worlds.cellular.arrays import neighborhood_keys
from tests.test_cellular_rule_inference import tiny_model_config


class RuleStressTest(unittest.TestCase):
    def test_augmented_training_resumes_both_rng_streams_exactly(self):
        config = RuleInferenceTrainingConfig(model=tiny_model_config(), max_steps=12, batch_size=2,
                                            train_rule_count=4, observation_noise=True, rule_change_probability=1.0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            full = train(config, root / "full", device="cpu")
            train(replace(config, max_steps=5), root / "resume", device="cpu")
            resumed = train(config, root / "resume", device="cpu", resume=True)
            model, payload = load_checkpoint(full)
            other, resumed_payload = load_checkpoint(resumed)
            self.assertEqual(payload["augmentation_rng_state"], resumed_payload["augmentation_rng_state"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
            clean = train(replace(config, observation_noise=False, rule_change_probability=0), root / "clean", device="cpu")
            _, clean_payload = load_checkpoint(clean)
            self.assertEqual(payload["numpy_rng_state"], clean_payload["numpy_rng_state"])

    def test_corruption_is_nested_and_only_touches_outputs(self):
        episode = sample_episodes(sample_rules(2, 7), np.random.default_rng(4),
                                  height=3, width=3, context_count=8)
        original = episode.support.copy()
        draws = np.random.default_rng(5).random(original[:, :, 1].shape)
        small = corrupt_outputs(original, draws, 0.1)
        large = corrupt_outputs(original, draws, 0.2)
        np.testing.assert_array_equal(original, episode.support)
        np.testing.assert_array_equal(large[:, :, 0], original[:, :, 0])
        self.assertFalse(((small != original) & (large == original)).any())
        np.testing.assert_array_equal(corrupt_outputs(original, draws, 0), original)
        np.testing.assert_array_equal(corrupt_outputs(original[:, :4], draws[:, :4], 0.2), large[:, :4])

    def test_change_splices_outputs_without_leaking_boundary_or_query(self):
        rules = sample_rules(2, 7)
        old = sample_episodes(rules[:1], np.random.default_rng(4), height=3, width=3, context_count=8)
        new = replace_context_rules(old, rules[1:])
        for count in (0, 1, 4, 8):
            actual = splice_context(old.support, new.support, count)
            np.testing.assert_array_equal(actual[:, :8-count], old.support[:, :8-count])
            np.testing.assert_array_equal(actual[:, 8-count:], new.support[:, 8-count:])
            np.testing.assert_array_equal(actual[:, :, 0], old.support[:, :, 0])
            np.testing.assert_array_equal(old.query, new.query)
        with self.assertRaises(ValueError):
            splice_context(old.support, new.support, 9)

    def test_family_posterior_uses_repeated_evidence_and_known_noise(self):
        # All-dead boards are fixed by B0; a live center has key 9 once per board.
        query = np.zeros((1, 3, 3), dtype=np.int64)
        query[0, 1, 1] = 1
        support = np.stack([query, np.zeros_like(query)], axis=1)[:, None].repeat(3, axis=1)
        support[:, 1:, 1, 1, 1] = 1
        result = baseline_probabilities(support, query, 0.1)["family_bayes"]
        self.assertAlmostEqual(result[0, 1, 1], 0.9)
        self.assertTrue((result[neighborhood_keys(query) == 0] == 0).all())
        permuted = baseline_probabilities(support[:, ::-1], query, 0.1)["family_bayes"]
        np.testing.assert_allclose(result, permuted)
        empty = baseline_probabilities(support[:, :0], query, 0.1)["family_bayes"]
        self.assertEqual(empty[0, 1, 1], 0.5)

    def test_probability_metrics_penalize_confident_errors_and_handle_no_evidence(self):
        target = np.array([0, 1])
        affected = np.array([True, False])
        covered = np.array([False, False])
        uncertain = probability_metrics(np.array([0.5, 0.5]), target, affected, covered)
        wrong = probability_metrics(np.array([1.0, 0.0]), target, affected, covered)
        self.assertGreater(wrong["nll"], uncertain["nll"])
        self.assertGreater(wrong["brier"], uncertain["brier"])
        self.assertEqual(uncertain["identifiable_affected_cells"], 0)
        json.dumps(wrong, allow_nan=False)

    def test_evaluation_is_frozen_reproducible_and_excludes_rule_identities(self):
        config = RuleInferenceTrainingConfig(model=tiny_model_config(), max_steps=1,
                                            batch_size=2, train_rule_count=4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = train(config, root, device="cpu")
            before = checkpoint.read_bytes()
            first = evaluate_stress(checkpoint, rule_count=2, queries_per_rule=1)
            second = evaluate_stress(checkpoint, rule_count=2, queries_per_rule=1)
            self.assertEqual(first, second)
            self.assertEqual(checkpoint.read_bytes(), before)
            json.dumps(first, allow_nan=False)
            validation = root / "validation.json"
            validation.write_text(json.dumps(first))
            third = evaluate_stress(checkpoint, rule_count=2, queries_per_rule=1,
                                    exclude_rules_from=(validation,))
            self.assertFalse(any(r in first["eval_rules"] for r in third["eval_rules"]))
            row = first["change"][0]
            self.assertIsNone(row["methods"]["model"]["identifiable_affected_accuracy"])
            for row in first["change"]:
                if row["post_change_count"] == 8:
                    self.assertEqual(row["methods"]["model"], row["methods"]["model_stable"])


if __name__ == "__main__":
    unittest.main()
