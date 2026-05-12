from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from intrep.problems.shogi_policy_value.generated_data_cycle import ShogiOnlineReplayResult


class RunShogiOnlineReplayScriptTest(unittest.TestCase):
    def test_passes_cli_arguments_to_online_replay_config_and_prints_result(self) -> None:
        module = _load_script_module()
        result = ShogiOnlineReplayResult(
            run_dir=Path("/tmp/online"),
            initial_checkpoint=Path("source.pt"),
            final_checkpoint=Path("/tmp/online/cycle-0002/best-checkpoint.pt"),
            next_checkpoint="best",
            replay_capacity=8,
            experience_store_dir=None,
            replay_seed_data_selection=None,
            training_eval_data_selection=None,
            preloaded_examples=0,
            fixed_eval_examples=0,
            cycles=(),
        )
        run_replay = Mock(return_value=result)

        with (
            patch.object(module, "run_shogi_online_replay", run_replay),
            patch.object(module, "print") as print_,
        ):
            module.main(
                [
                    "--checkpoint",
                    "source.pt",
                    "--run-dir",
                    "online",
                    "--cycles",
                    "2",
                    "--replay-capacity",
                    "8",
                    "--replay-sample-size",
                    "3",
                    "--min-replay-size",
                    "2",
                    "--experience-store-dir",
                    "data/shogi/experiences/online",
                    "--replay-seed-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--training-eval-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--games",
                    "4",
                    "--parallel-games",
                    "2",
                    "--board-backend",
                    "cshogi",
                    "--max-steps",
                    "5",
                    "--device",
                    "cuda",
                    "--seed",
                    "11",
                ]
            )

        config = run_replay.call_args.args[0]
        self.assertEqual(config.checkpoint, Path("source.pt"))
        self.assertEqual(config.run_dir, Path("online"))
        self.assertEqual(config.cycles, 2)
        self.assertEqual(config.replay_capacity, 8)
        self.assertEqual(config.replay_sample_size, 3)
        self.assertEqual(config.min_replay_size, 2)
        self.assertEqual(config.experience_store_dir, Path("data/shogi/experiences/online"))
        self.assertEqual(
            config.replay_seed_data_selection,
            Path("data/shogi/training-data-bundles/online/data-selection.json"),
        )
        self.assertEqual(
            config.training_eval_data_selection,
            Path("data/shogi/training-data-bundles/online/data-selection.json"),
        )
        self.assertEqual(config.games, 4)
        self.assertEqual(config.parallel_games, 2)
        self.assertEqual(config.board_backend, "cshogi")
        self.assertEqual(config.max_steps, 5)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.seed, 11)
        self.assertEqual(json.loads(print_.call_args.args[0]), result.to_json())


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_shogi_online_replay.py"
    spec = importlib.util.spec_from_file_location("run_shogi_online_replay", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
