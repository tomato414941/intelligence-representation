import tempfile
import unittest
from pathlib import Path

from intrep.shogi_game_record import load_shogi_move_choice_examples_from_usi_file, load_usi_move_games


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


if __name__ == "__main__":
    unittest.main()
