from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from intrep.problems.shogi_policy_value.online_replay import ShogiOnlineReplayResult


class RunShogiOnlineReplayScriptTest(unittest.TestCase):
    def test_uses_online_replay_parallelism_defaults(self) -> None:
        module = _load_script_module()
        result = ShogiOnlineReplayResult(
            run_dir=Path("/tmp/online"),
            initial_checkpoint=Path("source.pt"),
            final_checkpoint=Path("/tmp/online/iteration-0001/checkpoint.pt"),
            next_checkpoint="best",
            replay_capacity=8,
            experience_store_dir=None,
            replay_seed_data_selection=None,
            training_eval_data_selection=Path("eval/data-selection.json"),
            preloaded_examples=0,
            training_eval_examples=0,
            stop_reason=None,
            stopped_iteration_index=None,
            iterations=(),
        )
        run_replay = Mock(return_value=result)

        with (
            patch.object(module, "run_shogi_online_replay", run_replay),
            patch.object(module, "print"),
        ):
            module.main(
                [
                    "--checkpoint",
                    "source.pt",
                    "--run-dir",
                    "online",
                    "--training-eval-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--experience-source",
                    "checkpoint-self:4",
                ]
            )

        config = run_replay.call_args.args[0]
        self.assertEqual(config.training_budget.sampled_examples_per_iteration, 32768)
        self.assertEqual(config.generator_gate_worker_processes, 4)
        self.assertEqual(config.generation_worker_processes, 8)

    def test_passes_cli_arguments_to_online_replay_config_and_prints_result(self) -> None:
        module = _load_script_module()
        result = ShogiOnlineReplayResult(
            run_dir=Path("/tmp/online"),
            initial_checkpoint=Path("source.pt"),
            final_checkpoint=Path("/tmp/online/iteration-0002/best-checkpoint.pt"),
            next_checkpoint="best",
            replay_capacity=8,
            experience_store_dir=None,
            replay_seed_data_selection=None,
            training_eval_data_selection=Path("eval/data-selection.json"),
            preloaded_examples=0,
            training_eval_examples=0,
            stop_reason=None,
            stopped_iteration_index=None,
            iterations=(),
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
                    "--iterations",
                    "2",
                    "--replay-capacity",
                    "8",
                    "--sampled-examples-per-iteration",
                    "3",
                    "--min-replay-size",
                    "2",
                    "--training-batch-size",
                    "64",
                    "--target-sample-passes",
                    "2.5",
                    "--max-optimizer-steps-per-iteration",
                    "5",
                    "--generator-gate-games",
                    "6",
                    "--generator-gate-worker-processes",
                    "4",
                    "--experience-store-dir",
                    "data/shogi/experiences/online",
                    "--replay-seed-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--training-eval-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--experience-source",
                    "checkpoint-self:4",
                    "--checkpoint-move-selection-profile",
                    "self-play",
                    "--checkpoint-move-selection-temperature",
                    "0.75",
                    "--checkpoint-move-selection-temperature-plies",
                    "12",
                    "--concurrent-games-per-process",
                    "2",
                    "--generation-worker-processes",
                    "3",
                    "--generation-progress-every-plies",
                    "16",
                    "--board-backend",
                    "cshogi",
                    "--learning-rate",
                    "0.01",
                    "--weight-decay",
                    "0.02",
                    "--policy-loss-weight",
                    "0.7",
                    "--value-loss-weight",
                    "0.3",
                    "--allow-nonstandard-loss-weights",
                    "--device",
                    "cuda",
                    "--max-train-eval-examples",
                    "100",
                    "--max-eval-examples",
                    "50",
                    "--log-every",
                    "10",
                    "--seed",
                    "11",
                    "--num-workers",
                    "2",
                    "--pin-memory",
                    "--progress-every",
                    "20",
                    "--eval-every",
                    "25",
                    "--early-stopping-patience",
                    "3",
                ]
            )

        config = run_replay.call_args.args[0]
        self.assertEqual(config.checkpoint, Path("source.pt"))
        self.assertEqual(config.run_dir, Path("online"))
        self.assertEqual(config.iterations, 2)
        self.assertEqual(config.replay_capacity, 8)
        self.assertEqual(config.min_replay_size, 2)
        self.assertEqual(config.training_budget.sampled_examples_per_iteration, 3)
        self.assertEqual(config.training_budget.batch_size, 64)
        self.assertEqual(config.training_budget.target_sample_passes, 2.5)
        self.assertEqual(config.training_budget.max_optimizer_steps, 5)
        self.assertEqual(config.generator_gate_games, 6)
        self.assertEqual(config.generator_gate_worker_processes, 4)
        self.assertEqual(config.experience_store_dir, Path("data/shogi/experiences/online"))
        self.assertEqual(
            config.replay_seed_data_selection,
            Path("data/shogi/training-data-bundles/online/data-selection.json"),
        )
        self.assertEqual(
            config.training_eval_data_selection,
            Path("data/shogi/training-data-bundles/online/data-selection.json"),
        )
        self.assertEqual(len(config.experience_sources), 1)
        self.assertEqual(config.experience_sources[0].games, 4)
        self.assertEqual(config.experience_sources[0].black_player.kind, "checkpoint")
        self.assertEqual(config.experience_sources[0].white_player.kind, "checkpoint")
        self.assertEqual(config.experience_sources[0].black_player.move_selection_profile, "self-play")
        self.assertEqual(config.experience_sources[0].black_player.move_selection_temperature, 0.75)
        self.assertEqual(config.experience_sources[0].black_player.move_selection_temperature_plies, 12)
        self.assertEqual(config.concurrent_games_per_process, 2)
        self.assertEqual(config.generation_worker_processes, 3)
        self.assertEqual(config.generation_progress_every_plies, 16)
        self.assertEqual(config.board_backend, "cshogi")
        self.assertEqual(config.training_config.learning_rate, 0.01)
        self.assertEqual(config.training_config.weight_decay, 0.02)
        self.assertEqual(config.training_config.policy_loss_weight, 0.7)
        self.assertEqual(config.training_config.value_loss_weight, 0.3)
        self.assertTrue(config.training_config.allow_nonstandard_loss_weights)
        self.assertEqual(config.training_config.device, "cuda")
        self.assertEqual(config.training_config.max_train_eval_examples, 100)
        self.assertEqual(config.training_config.max_eval_examples, 50)
        self.assertEqual(config.training_config.log_every, 10)
        self.assertEqual(config.training_config.num_workers, 2)
        self.assertTrue(config.training_config.pin_memory)
        self.assertEqual(config.training_config.progress_every, 20)
        self.assertEqual(config.training_config.eval_every, 25)
        self.assertEqual(config.training_config.early_stopping_patience, 3)
        self.assertEqual(config.seed, 11)
        self.assertEqual(json.loads(print_.call_args.args[0]), result.to_json())

    def test_passes_multiple_experience_sources(self) -> None:
        module = _load_script_module()
        result = ShogiOnlineReplayResult(
            run_dir=Path("/tmp/online"),
            initial_checkpoint=Path("source.pt"),
            final_checkpoint=Path("/tmp/online/iteration-0001/checkpoint.pt"),
            next_checkpoint="final",
            replay_capacity=8,
            experience_store_dir=None,
            replay_seed_data_selection=None,
            training_eval_data_selection=Path("eval/data-selection.json"),
            preloaded_examples=0,
            training_eval_examples=0,
            stop_reason=None,
            stopped_iteration_index=None,
            iterations=(),
        )
        run_replay = Mock(return_value=result)

        with (
            patch.object(module, "run_shogi_online_replay", run_replay),
            patch.object(module, "print"),
        ):
            module.main(
                [
                    "--checkpoint",
                    "source.pt",
                    "--run-dir",
                    "online",
                    "--training-eval-data-selection",
                    "data/shogi/training-data-bundles/online/data-selection.json",
                    "--experience-source",
                    "checkpoint-self:2",
                    "--experience-source",
                    "checkpoint-vs-usi-balanced:3",
                    "--usi-command",
                    "engine",
                    "--usi-option",
                    "Threads=2",
                    "--usi-go-command",
                    "go nodes 4",
                    "--usi-read-timeout-seconds",
                    "31",
                ]
            )

        sources = run_replay.call_args.args[0].experience_sources
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0].games, 2)
        self.assertEqual(sources[0].black_player.kind, "checkpoint")
        self.assertEqual(sources[0].white_player.kind, "checkpoint")
        self.assertEqual(sources[1].games, 2)
        self.assertEqual(sources[1].black_player.kind, "checkpoint")
        self.assertEqual(sources[1].white_player.kind, "usi_engine")
        self.assertEqual(sources[1].white_player.usi_command, "engine")
        self.assertEqual(sources[1].white_player.usi_options, ("Threads=2",))
        self.assertEqual(sources[1].white_player.usi_go_command, "go nodes 4")
        self.assertEqual(sources[1].white_player.usi_read_timeout_seconds, 31)
        self.assertEqual(sources[2].games, 1)
        self.assertEqual(sources[2].black_player.kind, "usi_engine")
        self.assertEqual(sources[2].white_player.kind, "checkpoint")


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
