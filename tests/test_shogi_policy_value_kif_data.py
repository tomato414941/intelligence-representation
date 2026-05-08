import tempfile
import unittest
from pathlib import Path

from intrep.problems.shogi_policy_value.data import load_shogi_move_choice_examples_from_kif_file


class ShogiPolicyValueKifDataTest(unittest.TestCase):
    def test_loads_move_choice_examples_from_kif_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.kif"
            path.write_text(_sample_kif_text(), encoding="cp932")

            examples = load_shogi_move_choice_examples_from_kif_file(path)

        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])


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
