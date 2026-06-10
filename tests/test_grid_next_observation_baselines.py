import unittest

import torch

from intrep.domains.grid.encoding import (
    GRID_CELL_CLASSES,
    grid_observation_to_cell_class_ids,
    grid_position_to_cell_id,
)
from intrep.domains.grid.world import GridWorldState, Position, generate_grid_world_transition_table
from intrep.problems.grid_step_prediction.baselines import (
    copy_baseline,
    fit_per_cell_majority,
    naive_action_apply_baseline,
    per_cell_majority_baseline,
)
from intrep.problems.grid_step_prediction.metrics import next_observation_metrics


def _examples():
    return generate_grid_world_transition_table(
        GridWorldState(
            width=3,
            height=2,
            agent=Position(row=0, col=0),
            goal=Position(row=1, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )
    )


class GridCellClassEncodingTest(unittest.TestCase):
    def test_encodes_observation_cells_as_class_ids(self) -> None:
        examples = _examples()
        class_ids = grid_observation_to_cell_class_ids(examples[0].observation)

        self.assertEqual(GRID_CELL_CLASSES, ("empty", "agent", "goal", "wall"))
        self.assertEqual(class_ids.shape, (6,))
        self.assertEqual(class_ids[0].item(), 1)  # agent at (0, 0)
        self.assertEqual(class_ids[4].item(), 3)  # wall at (1, 1)
        self.assertEqual(class_ids[5].item(), 2)  # goal at (1, 2)


class NextObservationMetricsTest(unittest.TestCase):
    def test_perfect_prediction_scores_one_everywhere(self) -> None:
        examples = _examples()
        targets = torch.stack(
            [grid_observation_to_cell_class_ids(example.next_observation) for example in examples]
        )
        agent_scores = torch.zeros_like(targets, dtype=torch.float32)
        for index, example in enumerate(examples):
            agent_scores[index, grid_position_to_cell_id(example.next_observation.agent, width=3)] = 1.0

        metrics = next_observation_metrics(targets, agent_scores, examples, width=3)

        self.assertEqual(metrics.per_cell_accuracy, 1.0)
        self.assertEqual(metrics.changed_cell_accuracy, 1.0)
        self.assertEqual(metrics.whole_grid_match, 1.0)
        self.assertEqual(metrics.next_agent_cell_accuracy, 1.0)


class GridNextObservationBaselinesTest(unittest.TestCase):
    def test_copy_baseline_scores_stay_rate_on_agent_metric(self) -> None:
        examples = _examples()
        prediction = copy_baseline(examples, width=3)
        metrics = next_observation_metrics(prediction.class_ids, prediction.agent_scores, examples, width=3)

        stay_count = sum(
            1 for example in examples if example.next_observation.agent == example.observation.agent
        )
        self.assertAlmostEqual(metrics.next_agent_cell_accuracy, stay_count / len(examples))
        self.assertEqual(metrics.changed_cell_accuracy, 0.0)
        # 8 move transitions change 2 of 6 cells each: copy still looks high.
        self.assertAlmostEqual(metrics.per_cell_accuracy, 134 / 150, places=5)

    def test_naive_action_apply_fails_only_on_wall_blocks(self) -> None:
        examples = _examples()
        prediction = naive_action_apply_baseline(examples, width=3)
        metrics = next_observation_metrics(prediction.class_ids, prediction.agent_scores, examples, width=3)

        # Wall blocks: down from (0, 1), right from (1, 0), left from (1, 2).
        self.assertAlmostEqual(metrics.next_agent_cell_accuracy, 22 / 25)

    def test_per_cell_majority_predicts_static_structure(self) -> None:
        examples = _examples()
        table = fit_per_cell_majority(examples)
        prediction = per_cell_majority_baseline(table, examples)

        wall_cell = grid_position_to_cell_id(Position(row=1, col=1), width=3)
        self.assertEqual(table.majority_class_ids[wall_cell].item(), 3)
        self.assertEqual(prediction.class_ids.shape, (25, 6))
        self.assertEqual(prediction.agent_scores.shape, (25, 6))
        self.assertAlmostEqual(
            table.agent_frequencies.sum().item(),
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
