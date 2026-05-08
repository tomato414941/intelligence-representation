from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import shogi

from intrep.tasks.shogi_policy_value.dataset_definition import load_shogi_policy_value_dataset_definition
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(
    kind="checkpoint",
    name="black-model",
    settings={"checkpoint": "runs/shogi/model-a/checkpoint.pt", "policy": "mcts", "simulations": 8},
)
WHITE_ACTOR = ShogiActorSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})
YANEURAOU_ACTOR = ShogiActorSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 1"})


def _record(
    moves: tuple[str, ...],
    winner: str | None,
    *,
    black_actor: ShogiActorSpec = BLACK_ACTOR,
    white_actor: ShogiActorSpec = WHITE_ACTOR,
) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=black_actor,
        white_actor=white_actor,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
        end_reason="game_over",
    )


class ShogiExperienceStoreScriptsTest(unittest.TestCase):
    def test_appends_records_to_mutable_store(self) -> None:
        append_module = _load_script_module("append_shogi_experience_store")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "run-1.jsonl"
            second_input = root / "run-2.jsonl"
            store_dir = root / "data" / "shogi" / "experiences" / "main"
            first_records = [_record(("7g7f", "3c3d"), "black"), _record(("2g2f", "8c8d"), "white")]
            second_records = [_record(("5g5f", "5c5d"), "black")]
            write_shogi_game_records_jsonl(first_input, first_records)
            write_shogi_game_records_jsonl(second_input, second_records)

            first_result = append_module.append_shogi_experience_store(input_path=first_input, store_dir=store_dir)
            second_result = append_module.append_shogi_experience_store(input_path=second_input, store_dir=store_dir)

            self.assertEqual(first_result["added_games"], 2)
            self.assertEqual(second_result["added_games"], 1)
            self.assertEqual(load_shogi_game_records_jsonl(store_dir / "games.jsonl"), first_records + second_records)
            manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "shogi_experience_store_v1")
            self.assertEqual(manifest["game_count"], 3)
            self.assertEqual(manifest["position_stats"]["transition_count"], 6)
            self.assertEqual(manifest["position_stats"]["unique_position_count"], 4)
            self.assertEqual(manifest["position_stats"]["duplicate_position_count"], 2)
            self.assertEqual(manifest["position_stats"]["max_position_repeat_count"], 3)
            self.assertEqual(manifest["actor_pair_counts"], {"checkpoint:yaneuraou": 3})
            self.assertEqual(
                manifest["checkpoint_actor_counts"],
                {"runs/shogi/model-a/checkpoint.pt | policy=mcts | simulations=8": 3},
            )
            history_lines = (store_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(history_lines), 2)
            second_history = json.loads(history_lines[1])
            self.assertEqual(second_history["added_actor_pair_counts"], {"checkpoint:yaneuraou": 1})
            self.assertEqual(
                second_history["added_checkpoint_actor_counts"],
                {"runs/shogi/model-a/checkpoint.pt | policy=mcts | simulations=8": 1},
            )
            self.assertEqual(second_history["total_games"], 3)
            self.assertEqual(second_history["total_position_stats"]["unique_position_count"], 4)

    def test_creates_fixed_training_view_from_explicit_train_eval_sources(self) -> None:
        view_module = _load_script_module("create_shogi_training_view")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-source.jsonl"
            eval_path = root / "eval-source.jsonl"
            output_root = root / "data" / "shogi" / "datasets"
            train_records = [
                _record(("7g7f", "3c3d"), "black"),
                _record(("5g5f", "5c5d"), "black", black_actor=YANEURAOU_ACTOR, white_actor=YANEURAOU_ACTOR),
            ]
            eval_records = [
                _record(("2g2f", "8c8d"), "white"),
                _record(("6g6f", "6c6d"), "white", black_actor=YANEURAOU_ACTOR, white_actor=YANEURAOU_ACTOR),
            ]
            write_shogi_game_records_jsonl(train_path, train_records)
            write_shogi_game_records_jsonl(eval_path, eval_records)

            result = view_module.create_shogi_training_view(
                train_games=train_path,
                eval_games=eval_path,
                name="main-view-0001",
                output_root=output_root,
                max_train_games=1,
                max_eval_games=1,
            )

            view_dir = output_root / "main-view-0001"
            self.assertEqual(result["training_view"], str(view_dir))
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "games.jsonl"), [train_records[0], eval_records[0]])
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "train-games.jsonl"), [train_records[0]])
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "eval-games.jsonl"), [eval_records[0]])
            definition = load_shogi_policy_value_dataset_definition(view_dir / "dataset.json")
            self.assertEqual(definition.name, "main-view-0001")
            self.assertEqual(definition.policy_target_source, "chosen_move")
            self.assertEqual(definition.policy_temperature_cp, 100.0)
            self.assertEqual(definition.policy_mate_cp, 100000.0)
            self.assertEqual(definition.value_target_source, "winner")
            self.assertEqual(definition.score_cp_scale, 600.0)
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-games.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-games.jsonl")
            self.assertEqual(definition.train_sources[0].max_games, 1)
            self.assertEqual(definition.eval_sources[0].max_games, 1)
            manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "shogi_training_view_v1")
            self.assertEqual(manifest["train_source_games_jsonl"], str(train_path))
            self.assertEqual(manifest["eval_source_games_jsonl"], str(eval_path))
            self.assertEqual(manifest["max_train_games"], 1)
            self.assertEqual(manifest["max_eval_games"], 1)
            self.assertEqual(manifest["actor_pair_counts"], {"checkpoint:yaneuraou": 2})
            self.assertEqual(manifest["train_actor_pair_counts"], {"checkpoint:yaneuraou": 1})
            self.assertEqual(manifest["eval_actor_pair_counts"], {"checkpoint:yaneuraou": 1})
            self.assertEqual(manifest["train_games"], 1)
            self.assertEqual(manifest["eval_games"], 1)
            self.assertEqual(manifest["train_position_stats"]["transition_count"], 2)
            self.assertEqual(manifest["eval_position_stats"]["transition_count"], 2)
            self.assertEqual(manifest["train_eval_position_overlap_count"], 1)
            self.assertEqual(manifest["position_stats"]["unique_position_count"], 3)
            self.assertEqual(manifest["policy_target_source"], "chosen_move")
            self.assertEqual(manifest["value_target_source"], "winner")

    def test_refuses_to_overwrite_existing_training_view(self) -> None:
        view_module = _load_script_module("create_shogi_training_view")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(
                train_path,
                [_record(("7g7f", "3c3d"), "black")],
            )
            write_shogi_game_records_jsonl(
                eval_path,
                [_record(("2g2f", "8c8d"), "white")],
            )
            (output_root / "existing").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                view_module.create_shogi_training_view(
                    train_games=train_path,
                    eval_games=eval_path,
                    name="existing",
                    output_root=output_root,
                )


def _load_script_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
