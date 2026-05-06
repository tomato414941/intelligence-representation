import json
import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.tasks.shogi_move_choice.data import load_shogi_move_choice_examples_from_game_records_jsonl
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_record_to_json,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={"checkpoint": "black.pt"})
WHITE_ACTOR = ShogiActorSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )


class ShogiMoveChoiceDataTest(unittest.TestCase):
    def test_loads_move_choice_examples_from_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [_record(("7g7f", "3c3d"), "white")])

            examples = load_shogi_move_choice_examples_from_game_records_jsonl(path)

        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])
        self.assertEqual([example.value_target for example in examples], [-1.0, 1.0])
        self.assertEqual([example.game_index for example in examples], [0, 0])
        self.assertEqual([example.ply_index for example in examples], [0, 1])

    def test_loads_move_choice_examples_from_game_record_jsonl_text(self) -> None:
        record = _record(("7g7f", "3c3d"), "black")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            records = load_shogi_move_choice_examples_from_game_records_jsonl(path)

        self.assertEqual([record.chosen_move for record in records], ["7g7f", "3c3d"])
        self.assertEqual([record.value_target for record in records], [1.0, -1.0])
        self.assertEqual([record.game_index for record in records], [0, 0])
        self.assertEqual([record.ply_index for record in records], [0, 1])


if __name__ == "__main__":
    unittest.main()
