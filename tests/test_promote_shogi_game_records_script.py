from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import shogi

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


class PromoteShogiGameRecordsScriptTest(unittest.TestCase):
    def test_promotes_records_into_collection_with_split_and_manifest(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "run" / "generated-games.jsonl"
            records = (
                _record(("7g7f", "3c3d"), "black"),
                _record(("2g2f", "8c8d"), "white"),
                _record(("5g5f", "5c5d"), "black"),
                _record(("6g6f", "6c6d"), "white"),
            )
            write_shogi_game_records_jsonl(input_path, records)

            result = module.promote_shogi_game_records(
                input_path=input_path,
                name="sample-records",
                output_root=root / "data" / "shogi" / "records",
                eval_ratio=0.25,
            )

            collection_dir = root / "data" / "shogi" / "records" / "sample-records"
            self.assertEqual(result["record_collection"], str(collection_dir))
            self.assertEqual(load_shogi_game_records_jsonl(collection_dir / "games.jsonl"), list(records))
            self.assertEqual(load_shogi_game_records_jsonl(collection_dir / "train-games.jsonl"), list(records[:3]))
            self.assertEqual(load_shogi_game_records_jsonl(collection_dir / "eval-games.jsonl"), list(records[3:]))
            manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "shogi_game_record_collection_v1")
            self.assertEqual(manifest["record_schema"], "shogi_game_record_jsonl")
            self.assertEqual(manifest["name"], "sample-records")
            self.assertEqual(manifest["source_path"], str(input_path))
            self.assertEqual(manifest["game_count"], 4)
            self.assertEqual(manifest["transition_count"], 8)
            self.assertEqual(manifest["train_games"], 3)
            self.assertEqual(manifest["eval_games"], 1)
            self.assertEqual(manifest["files"]["games"], "games.jsonl")

    def test_refuses_to_overwrite_existing_collection(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "generated-games.jsonl"
            write_shogi_game_records_jsonl(
                input_path,
                [_record(("7g7f", "3c3d"), "black"), _record(("2g2f", "8c8d"), "white")],
            )
            output_root = root / "records"
            (output_root / "existing").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                module.promote_shogi_game_records(
                    input_path=input_path,
                    name="existing",
                    output_root=output_root,
                    eval_ratio=0.25,
                )


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "promote_shogi_game_records.py"
    spec = importlib.util.spec_from_file_location("promote_shogi_game_records", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
