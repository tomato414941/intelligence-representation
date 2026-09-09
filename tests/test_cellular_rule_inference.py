import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from intrep.problems.cellular_rule_inference.episodes import (
    evidence_coverage,
    replace_context_rules,
    rule_id,
    sample_episodes,
    sample_rules,
)
from intrep.problems.cellular_rule_inference.evaluate import context_baselines, evaluate
from intrep.problems.cellular_rule_inference.training import (
    RuleInferenceTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.representation.assemblies.cellular_rule_inference import (
    CellularRuleInferenceModel,
    CellularRuleInferenceModelConfig,
)
from intrep.worlds.cellular.arrays import step_grids
from intrep.worlds.cellular.world import (
    LIFE_RULE,
    CellularRule,
    CellularWorldState,
    step_cellular_state,
)


def tiny_model_config():
    return CellularRuleInferenceModelConfig(height=3, width=3, embedding_dim=8, hidden_dim=16, num_heads=2, num_layers=1)


class RuleEpisodesTest(unittest.TestCase):
    def test_batch_truth_matches_existing_world_for_random_rules_and_borders(self):
        rules = [LIFE_RULE, *sample_rules(6, 7)]
        states = np.random.default_rng(17).integers(2, size=(7, 3, 4, 5))
        actual = step_grids(states, rules)
        for index, rule in enumerate(rules):
            for board_index, grid in enumerate(states[index]):
                state = CellularWorldState(5, 4, tuple(tuple(int(c) for c in row) for row in grid))
                expected = step_cellular_state(state, rule)
                np.testing.assert_array_equal(actual[index, board_index], expected.grid)

    def test_disjoint_rules_use_identity_even_when_seeds_overlap(self):
        training = sample_rules(20, 7)
        testing = sample_rules(20, 7, excluded=tuple(training))
        self.assertEqual(len({rule_id(r) for r in testing}), 20)
        self.assertFalse({rule_id(r) for r in training} & {rule_id(r) for r in testing})

    def test_queries_are_not_demonstrated_and_generation_is_reproducible(self):
        rules = sample_rules(10, 9)
        kwargs = {"height": 3, "width": 3, "context_count": 8}
        a = sample_episodes(rules, np.random.default_rng(4), **kwargs)
        b = sample_episodes(rules, np.random.default_rng(4), **kwargs)
        np.testing.assert_array_equal(a.support, b.support)
        self.assertFalse(np.any(np.all(a.support[:, :, 0] == a.query[:, None], axis=(-2, -1))))

    def test_coverage_is_monotonic_and_does_not_read_answers(self):
        episodes = sample_episodes(sample_rules(10, 9), np.random.default_rng(4), height=4, width=4, context_count=8)
        previous = np.zeros_like(episodes.query, dtype=bool)
        for count in (0, 1, 4, 8):
            context = episodes.support[:, :count]
            covered = evidence_coverage(context, episodes.query)
            self.assertFalse((previous & ~covered).any())
            flipped = context.copy()
            flipped[:, :, 1] = 1 - flipped[:, :, 1]
            np.testing.assert_array_equal(covered, evidence_coverage(flipped, episodes.query))
            previous = covered

    def test_counterfactual_keeps_inputs_and_changes_only_rule_outputs(self):
        empty = CellularRule(frozenset(), frozenset())
        full = CellularRule(frozenset(range(9)), frozenset(range(9)))
        a = sample_episodes([empty], np.random.default_rng(4), height=3, width=3, context_count=8)
        b = replace_context_rules(a, [full])
        np.testing.assert_array_equal(a.query, b.query)
        np.testing.assert_array_equal(a.support[:, :, 0], b.support[:, :, 0])
        self.assertTrue((a.targets == 0).all() and (b.targets == 1).all())
        self.assertTrue((a.support[:, :, 1] == 0).all() and (b.support[:, :, 1] == 1).all())

    def test_family_lookup_is_exact_where_evidence_covers_query(self):
        episodes = sample_episodes(sample_rules(10, 9), np.random.default_rng(4), height=4, width=4, context_count=8)
        covered = evidence_coverage(episodes.support, episodes.query)
        _, lookup = context_baselines(episodes.support, episodes.query)
        self.assertTrue(covered.any())
        np.testing.assert_array_equal(lookup[covered], episodes.targets[covered])


class RuleInferenceModelTest(unittest.TestCase):
    def test_variable_contexts_output_query_cells_and_propagate_context_gradients(self):
        model = CellularRuleInferenceModel(tiny_model_config())
        query = torch.rand(2, 3, 3)
        for count in (0, 1, 4, 8):
            support = torch.rand(2, count, 2, 3, 3, requires_grad=True)
            logits = model(support, query)
            self.assertEqual(tuple(logits.shape), (2, 9, 2))
            logits.sum().backward()
            if count:
                self.assertGreater(float(support.grad.abs().sum()), 0)

    def test_resume_matches_uninterrupted_training_and_evaluation_does_not_learn(self):
        cfg = RuleInferenceTrainingConfig(model=tiny_model_config(), max_steps=2, batch_size=2, train_rule_count=4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            full = train(cfg, root / 'full', device='cpu')
            train(replace(cfg, max_steps=1), root / 'resumed', device='cpu')
            resumed = train(cfg, root / 'resumed', device='cpu', resume=True)
            model, _ = load_checkpoint(full)
            resumed_model, _ = load_checkpoint(resumed)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            before = resumed.read_bytes()
            result = evaluate(resumed, rule_count=2, queries_per_rule=2)
            self.assertEqual(before, resumed.read_bytes())
            self.assertEqual([r['context_count'] for r in result['summaries']], [0, 1, 4, 8])
            zero = result['summaries'][0]
            self.assertEqual(zero['correct_context']['accuracy'], zero['wrong_context']['accuracy'])
            self.assertIsNone(zero['correct_context']['covered_accuracy'])
            self.assertEqual(zero['prediction_change_rate'], 0)
            json.dumps(result, allow_nan=False)
            validation = root / 'validation.json'
            validation.write_text(json.dumps(result))
            test = evaluate(resumed, rule_count=2, queries_per_rule=1, exclude_rules_from=validation)
            self.assertFalse({row['rule_id'] for row in result['per_rule'][0]} & {row['rule_id'] for row in test['per_rule'][0]})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(replace(cfg, train_rule_count=5), root / 'resumed', resume=True)


if __name__ == '__main__':
    unittest.main()
