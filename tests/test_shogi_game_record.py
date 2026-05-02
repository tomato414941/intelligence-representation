import tempfile
import unittest
from pathlib import Path

from intrep.shogi_game_record import (
    convert_kif_files_to_usi_file,
    load_kif_game,
    load_shogi_move_choice_examples_from_kif_file,
    load_shogi_move_choice_examples_from_usi_file,
    load_usi_move_games,
    write_usi_move_games,
)


class ShogiGameRecordTest(unittest.TestCase):
    def test_loads_usi_move_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.txt"
            path.write_text("# comment\n7g7f 3c3d\n\n2g2f 8c8d\n", encoding="utf-8")

            games = load_usi_move_games(path)

        self.assertEqual(games, [("7g7f", "3c3d"), ("2g2f", "8c8d")])

    def test_loads_move_choice_examples_from_usi_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.txt"
            path.write_text("7g7f 3c3d\n", encoding="utf-8")

            examples = load_shogi_move_choice_examples_from_usi_file(path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].chosen_move, "7g7f")
        self.assertEqual(examples[1].chosen_move, "3c3d")

    def test_loads_kif_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.kif"
            path.write_text(
                "\n".join(
                    [
                        "開始日時：2026/05/02",
                        "手合割：平手",
                        "先手：black",
                        "後手：white",
                        "手数----指手---------消費時間--",
                        "   1 ７六歩(77)        ( 0:00/00:00:00)",
                        "   2 ３四歩(33)        ( 0:00/00:00:00)",
                    ]
                )
                + "\n",
                encoding="cp932",
            )

            moves = load_kif_game(path)
            examples = load_shogi_move_choice_examples_from_kif_file(path)

        self.assertEqual(moves, ("7g7f", "3c3d"))
        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])

    def test_writes_usi_move_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.txt"

            write_usi_move_games(path, [("7g7f", "3c3d"), ("2g2f",)])

            games = load_usi_move_games(path)

        self.assertEqual(games, [("7g7f", "3c3d"), ("2g2f",)])

    def test_converts_kif_files_to_usi_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kif_path = root / "game.kif"
            kif_path.write_text(
                "\n".join(
                    [
                        "開始日時：2026/05/02",
                        "手合割：平手",
                        "先手：black",
                        "後手：white",
                        "手数----指手---------消費時間--",
                        "   1 ７六歩(77)        ( 0:00/00:00:00)",
                        "   2 ３四歩(33)        ( 0:00/00:00:00)",
                    ]
                )
                + "\n",
                encoding="cp932",
            )
            output_path = root / "games.txt"

            count = convert_kif_files_to_usi_file([kif_path], output_path)

            self.assertEqual(count, 1)
            self.assertEqual(load_usi_move_games(output_path), [("7g7f", "3c3d")])


if __name__ == "__main__":
    unittest.main()
