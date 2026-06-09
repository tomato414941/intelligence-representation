import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.domains.grid.world import GridWorldState, Position
from intrep.problems.grid_step_prediction import diagnose_heldout
from intrep.problems.grid_step_prediction.heldout_diagnostics import run_held_out_cell_sweep
from intrep.problems.grid_step_prediction.training import GridStepPredictionConfig


def _tiny_config() -> GridStepPredictionConfig:
    return GridStepPredictionConfig(
        max_steps=1,
        batch_size=5,
        embedding_dim=16,
        num_heads=2,
        hidden_dim=32,
        num_layers=1,
        device="cpu",
    )


class GridHeldOutDiagnosticsTest(unittest.TestCase):
    def test_sweep_covers_every_agent_cell_and_seed(self) -> None:
        state = GridWorldState(
            width=3,
            height=2,
            agent=Position(row=0, col=0),
            goal=Position(row=1, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )

        runs = run_held_out_cell_sweep(state, seeds=(1, 2), config=_tiny_config())

        held_out_cells = {run.held_out_cell for run in runs}
        self.assertEqual(len(runs), 10)
        self.assertEqual(
            held_out_cells,
            {
                Position(row=0, col=0),
                Position(row=0, col=1),
                Position(row=0, col=2),
                Position(row=1, col=0),
                Position(row=1, col=2),
            },
        )
        for run in runs:
            self.assertEqual(run.train_case_count, 20)
            self.assertEqual(run.eval_case_count, 5)
            self.assertEqual(len(run.predictions), 5)
            for prediction in run.predictions:
                self.assertEqual(prediction.agent, run.held_out_cell)
                self.assertEqual(prediction.correct, prediction.predicted_next == prediction.true_next)

    def test_cli_writes_diagnostic_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "grid-heldout-diagnostic.json"

            with redirect_stdout(output):
                diagnose_heldout.main(
                    [
                        "--metrics-path",
                        str(metrics_path),
                        "--seeds",
                        "1",
                        "--max-steps",
                        "1",
                        "--embedding-dim",
                        "16",
                        "--num-heads",
                        "2",
                        "--hidden-dim",
                        "32",
                        "--num-layers",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep diagnose grid held-out generalization", output.getvalue())
        self.assertEqual(payload["schema_version"], "intrep.grid_step_heldout_diagnostic.v1")
        self.assertEqual(payload["seeds"], [1])
        self.assertEqual(len(payload["runs"]), 5)
        first_run = payload["runs"][0]
        self.assertEqual(first_run["train_case_count"], 20)
        self.assertEqual(first_run["eval_case_count"], 5)
        self.assertEqual(len(first_run["predictions"]), 5)
        self.assertIn("predicted_next", first_run["predictions"][0])


if __name__ == "__main__":
    unittest.main()
