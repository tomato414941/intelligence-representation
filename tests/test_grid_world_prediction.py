import unittest

import torch

from intrep.grid_world import GridWorldState, Position, generate_grid_world_experience, generate_grid_world_transition_table
from intrep.grid_world_prediction import (
    GridNextCellDataset,
    GridNextCellPredictionConfig,
    GridNextCellPredictor,
    train_grid_next_cell_predictor,
)


class GridWorldPredictionTest(unittest.TestCase):
    def test_dataset_returns_grid_tensor_action_id_and_next_cell_target(self) -> None:
        examples = generate_grid_world_experience(
            actions=("right",),
            initial_state=GridWorldState(
                width=3,
                height=2,
                agent=Position(row=0, col=0),
                goal=Position(row=1, col=2),
            ),
        )

        dataset = GridNextCellDataset(examples)
        observation, action_id, next_cell_id = dataset[0]

        self.assertEqual(tuple(observation.shape), (3, 2, 3))
        self.assertEqual(int(action_id.item()), 3)
        self.assertEqual(int(next_cell_id.item()), 1)

    def test_predictor_returns_next_cell_logits(self) -> None:
        model = GridNextCellPredictor(
            height=2,
            width=3,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )

        logits = model(torch.zeros((2, 3, 2, 3)), torch.tensor([3, 4], dtype=torch.long))

        self.assertEqual(tuple(logits.shape), (2, 6))

    def test_training_runs_on_grid_experience(self) -> None:
        examples = generate_grid_world_transition_table(
            GridWorldState(
                width=3,
                height=2,
                agent=Position(row=0, col=0),
                goal=Position(row=1, col=2),
                walls=frozenset({Position(row=1, col=1)}),
            )
        )

        result = train_grid_next_cell_predictor(
            examples,
            config=GridNextCellPredictionConfig(
                max_steps=200,
                batch_size=5,
                learning_rate=0.01,
                seed=31,
                embedding_dim=32,
                num_heads=2,
                hidden_dim=64,
                num_layers=1,
                device="cpu",
            ),
        )

        self.assertEqual(result.train_case_count, 25)
        self.assertGreater(result.initial_loss, 0.0)
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertGreaterEqual(result.accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()
