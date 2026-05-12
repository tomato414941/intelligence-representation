from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from intrep.problems.shogi_policy_value.generated_data_cycle import ShogiGeneratedDataTrainingLoopResult


class RunShogiGeneratedDataTrainingLoopScriptTest(unittest.TestCase):
    def test_passes_cli_arguments_to_loop_config_and_prints_result(self) -> None:
        module = _load_script_module()
        result = ShogiGeneratedDataTrainingLoopResult(
            run_dir=Path("/tmp/loop"),
            initial_checkpoint=Path("source.pt"),
            final_checkpoint=Path("/tmp/loop/cycle-0002/best-checkpoint.pt"),
            next_checkpoint="final",
            cycles=(),
        )
        run_loop = Mock(return_value=result)

        with (
            patch.object(module, "run_shogi_generated_data_training_loop", run_loop),
            patch.object(module, "print") as print_,
        ):
            module.main(
                [
                    "--checkpoint",
                    "source.pt",
                    "--run-dir",
                    "loop",
                    "--cycles",
                    "2",
                    "--next-checkpoint",
                    "final",
                    "--games",
                    "3",
                    "--parallel-games",
                    "2",
                    "--generation-worker-processes",
                    "3",
                    "--seed",
                    "11",
                    "--board-backend",
                    "cshogi",
                    "--max-steps",
                    "4",
                ]
            )

        config = run_loop.call_args.args[0]
        self.assertEqual(config.checkpoint, Path("source.pt"))
        self.assertEqual(config.run_dir, Path("loop"))
        self.assertEqual(config.cycles, 2)
        self.assertEqual(config.next_checkpoint, "final")
        self.assertEqual(config.games, 3)
        self.assertEqual(config.parallel_games, 2)
        self.assertEqual(config.generation_worker_processes, 3)
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.board_backend, "cshogi")
        self.assertEqual(config.max_steps, 4)
        self.assertEqual(json.loads(print_.call_args.args[0]), result.to_json())


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_generated_data_training_loop.py"
    spec = importlib.util.spec_from_file_location("run_shogi_generated_data_training_loop", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
