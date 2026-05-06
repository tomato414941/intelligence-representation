from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import shogi

from intrep.tasks.shogi_move_choice.dataset_definition import load_shogi_move_choice_dataset_definition
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
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
            history_lines = (store_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(history_lines), 2)
            self.assertEqual(json.loads(history_lines[1])["total_games"], 3)

    def test_creates_fixed_training_view_from_store(self) -> None:
        append_module = _load_script_module("append_shogi_experience_store")
        view_module = _load_script_module("create_shogi_training_view")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "generated-games.jsonl"
            store_dir = root / "data" / "shogi" / "experiences" / "main"
            output_root = root / "data" / "shogi" / "datasets"
            records = [
                _record(("7g7f", "3c3d"), "black"),
                _record(("2g2f", "8c8d"), "white"),
                _record(("5g5f", "5c5d"), "black"),
                _record(("6g6f", "6c6d"), "white"),
            ]
            write_shogi_game_records_jsonl(input_path, records)
            append_module.append_shogi_experience_store(input_path=input_path, store_dir=store_dir)

            result = view_module.create_shogi_training_view(
                store_dir=store_dir,
                name="main-view-0001",
                output_root=output_root,
                eval_ratio=0.25,
            )

            view_dir = output_root / "main-view-0001"
            self.assertEqual(result["training_view"], str(view_dir))
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "games.jsonl"), records)
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "train-games.jsonl"), records[:3])
            self.assertEqual(load_shogi_game_records_jsonl(view_dir / "eval-games.jsonl"), records[3:])
            definition = load_shogi_move_choice_dataset_definition(view_dir / "dataset.json")
            self.assertEqual(definition.name, "main-view-0001")
            self.assertEqual(definition.train_sources[0].path, view_dir / "train-games.jsonl")
            self.assertEqual(definition.eval_sources[0].path, view_dir / "eval-games.jsonl")
            manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "shogi_training_view_v1")
            self.assertEqual(manifest["store"], str(store_dir))

    def test_refuses_to_overwrite_existing_training_view(self) -> None:
        view_module = _load_script_module("create_shogi_training_view")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store_dir = root / "store"
            output_root = root / "datasets"
            write_shogi_game_records_jsonl(
                store_dir / "games.jsonl",
                [_record(("7g7f", "3c3d"), "black"), _record(("2g2f", "8c8d"), "white")],
            )
            (output_root / "existing").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                view_module.create_shogi_training_view(
                    store_dir=store_dir,
                    name="existing",
                    output_root=output_root,
                    eval_ratio=0.25,
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
