import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from intrep.problems.cellular_rule_inference.control import (
    CONTEXT_COUNTS,
    METHODS,
    ROUNDS,
    choose_actions,
    decision_metrics,
    evaluate_control,
    expected_matches,
    intervention_boards,
    predict_candidates,
    run_tasks,
    sample_tasks,
    summarize_round,
)
from intrep.problems.cellular_rule_inference.episodes import sample_rules
from intrep.problems.cellular_rule_inference.training import (
    RuleInferenceTrainingConfig,
    train,
)
from intrep.representation.assemblies.cellular_rule_inference import (
    CellularRuleInferenceModel,
)
from intrep.worlds.cellular.arrays import step_grids
from tests.test_cellular_rule_inference import tiny_model_config


class CellularControlTest(unittest.TestCase):
    def test_interventions_flip_exactly_one_cell_without_mutating_the_input(self):
        board = np.random.default_rng(2).integers(2, size=(2, 3, 4))
        original = board.copy()
        actions = intervention_boards(board)
        np.testing.assert_array_equal(board, original)
        np.testing.assert_array_equal(actions[:, 0], board)
        np.testing.assert_array_equal((actions[:, 1:] != board[:, None]).sum(axis=(-2, -1)), 1)
        for index in range(12):
            expected = board.copy()
            expected[:, index // 4, index % 4] ^= 1
            np.testing.assert_array_equal(actions[:, index + 1], expected)

    def test_task_candidates_never_repeat_across_rounds_and_do_not_depend_on_a_rule(self):
        first = sample_tasks(np.random.default_rng(42), trials=2, height=3, width=3)
        second = sample_tasks(np.random.default_rng(42), trials=2, height=3, width=3)
        for name in ("boards", "goals", "tie_orders"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        for trial in range(2):
            pools = intervention_boards(first.boards[:, trial])
            flattened = pools.reshape(-1, 9)
            self.assertEqual(len(np.unique(flattened, axis=0)), ROUNDS * 10)
            for order in first.tie_orders[:, trial]:
                np.testing.assert_array_equal(np.sort(order), np.arange(10))

    def test_action_selection_uses_probabilities_and_shared_tie_breaking(self):
        # Both candidates threshold to the same board, but their expected utility differs.
        probability = np.array([[[[0.51, 0.49]], [[0.9, 0.1]], [[0.9, 0.1]]]])
        goal = np.array([[[1, 0]]])
        scores = expected_matches(probability, goal)
        self.assertEqual(choose_actions(scores, np.array([[2, 1, 0]])).tolist(), [2])
        self.assertGreater(scores[0, 1], scores[0, 0])

    def test_model_candidate_batching_preserves_task_and_context_alignment(self):
        torch.manual_seed(8)
        model = CellularRuleInferenceModel(tiny_model_config()).eval()
        rng = np.random.default_rng(7)
        support = rng.integers(2, size=(2, 4, 2, 3, 3))
        candidates = intervention_boards(rng.integers(2, size=(2, 3, 3)))
        small = predict_candidates(model, support, candidates, 3)
        large = predict_candidates(model, support, candidates, 64)
        np.testing.assert_allclose(small, large, rtol=1e-5, atol=1e-6)

    def test_only_executed_transitions_are_carried_into_later_decisions(self):
        tasks = sample_tasks(np.random.default_rng(12), trials=2, height=3, width=3)
        rule, donor = sample_rules(2, 15)
        seen = []

        def spy(model, support, candidates, batch_size):
            seen.append((support.copy(), candidates.copy()))
            return np.full(candidates.shape, 0.5)

        model = CellularRuleInferenceModel(tiny_model_config())
        with patch("intrep.problems.cellular_rule_inference.control.predict_candidates", side_effect=spy):
            rows = run_tasks(model, rule, donor, tasks, retain_trace=True)
        history = []
        cursor = 0
        for turn in range(ROUNDS):
            count = max(k for k in CONTEXT_COUNTS if k <= turn)
            support, candidates = seen[cursor]
            cursor += 1 if turn == 0 else 3
            self.assertEqual(support.shape[1], count)
            if count:
                np.testing.assert_array_equal(support, np.stack(history[-count:], axis=1))
                for batch in range(2):
                    self.assertFalse(np.any(np.all(
                        support[batch, :, 0, None] == candidates[batch, None], axis=(-2, -1))))
            # Equal predictions force the independently sampled first tie-break action.
            actions = tasks.tie_orders[turn, :, 0]
            chosen = candidates[np.arange(2), actions]
            observed = step_grids(chosen, [rule] * 2)
            history.append(np.stack((chosen, observed), axis=1))
            self.assertEqual(rows[turn]["trace"]["experience_action"], actions[0])
            np.testing.assert_array_equal(rows[turn]["trace"]["observed"], observed[0])

    def test_metrics_remove_static_free_points_and_count_all_optimal_ties(self):
        scores = np.array([[0, 4, 4], [3, 3, 3]])
        choices = {name: np.array([0, 0]) for name in METHODS if name not in ("random", "oracle")}
        choices["forgetful"] = np.array([2, 0])
        result = decision_metrics(scores, choices)
        self.assertEqual(result["decision_tasks"], 1)
        self.assertEqual(result["methods"]["experience"]["regret_sum"], 4)
        self.assertEqual(result["methods"]["forgetful"]["optimal_count"], 1)
        self.assertAlmostEqual(result["methods"]["random"]["optimal_count"], 2 / 3)
        self.assertAlmostEqual(result["methods"]["random"]["regret_sum"], 4 / 3)
        self.assertEqual(result, decision_metrics(scores + 20, choices))
        empty = {"round": 0, "context_count": 0, **decision_metrics(scores[1:], {
            name: action[1:] for name, action in choices.items()})}
        summary = summarize_round([empty, empty])
        self.assertIsNone(summary["methods"]["experience"]["optimal_action_rate"]["mean"])
        json.dumps(summary, allow_nan=False)

    def test_evaluation_is_reproducible_frozen_and_excludes_prior_rule_identities(self):
        config = RuleInferenceTrainingConfig(model=tiny_model_config(), max_steps=1,
                                            batch_size=2, train_rule_count=4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = train(config, root, device="cpu")
            before = checkpoint.read_bytes()
            first = evaluate_control(checkpoint, rule_count=2, trials_per_rule=1)
            second = evaluate_control(checkpoint, rule_count=2, trials_per_rule=1)
            self.assertEqual(first, second)
            self.assertEqual(checkpoint.read_bytes(), before)
            json.dumps(first, allow_nan=False)
            validation = root / "validation.json"
            validation.write_text(json.dumps(first))
            third = evaluate_control(checkpoint, rule_count=2, trials_per_rule=1,
                                     exclude_rules_from=(validation,))
            self.assertFalse(any(rule in first["eval_rules"] for rule in third["eval_rules"]))
            for row in first["summaries"][:1]:
                self.assertEqual(row["methods"]["experience"], row["methods"]["forgetful"])
                self.assertEqual(row["methods"]["experience"], row["methods"]["wrong_context"])


if __name__ == "__main__":
    unittest.main()
