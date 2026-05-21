import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.domains.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_record_from_usi_moves,
)
from intrep.domains.shogi.kif_io import (
    convert_kif_files_to_game_records_jsonl,
    load_kif_game,
    load_kif_game_record,
)


class ShogiKifIoTest(unittest.TestCase):
    def test_loads_kif_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.kif"
            path.write_text(_sample_kif_text(), encoding="cp932")

            moves = load_kif_game(path)
            record_moves, winner = load_kif_game_record(path)

        self.assertEqual(moves, ("7g7f", "3c3d"))
        self.assertEqual(record_moves, ("7g7f", "3c3d"))
        self.assertEqual(winner, "w")

    def test_converts_kif_files_to_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kif_path = root / "game.kif"
            kif_path.write_text(_sample_kif_text(), encoding="cp932")
            output_path = root / "games.jsonl"

            count = convert_kif_files_to_game_records_jsonl([kif_path], output_path)

            self.assertEqual(count, 1)
            self.assertEqual(
                load_shogi_game_records_jsonl(output_path),
                [
                    shogi_game_record_from_usi_moves(
                        ("7g7f", "3c3d"),
                        black_actor=ShogiActorSpec(kind="kif", name="black", settings={}),
                        white_actor=ShogiActorSpec(kind="kif", name="white", settings={}),
                        initial_position_sfen=shogi.Board().sfen(),
                        winner="white",
                    )
                ],
            )


def _sample_kif_text() -> str:
    return (
        "\n".join(
            [
                "開始日時：2026/05/02",
                "手合割：平手",
                "先手：black",
                "後手：white",
                "手数----指手---------消費時間--",
                "   1 ７六歩(77)        ( 0:00/00:00:00)",
                "   2 ３四歩(33)        ( 0:00/00:00:00)",
                "   3 投了",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    unittest.main()
