import tempfile
import unittest
from pathlib import Path

from intrep.shogi.game_record import PlayerSpec, ShogiGameRecord, write_shogi_game_records_jsonl
from intrep.shogi.move_choice_data import load_shogi_move_choice_examples_from_game_records_jsonl


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={"checkpoint": "black.pt"})
WHITE_PLAYER = PlayerSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


class ShogiMoveChoiceDataTest(unittest.TestCase):
    def test_loads_move_choice_examples_from_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(
                path,
                [
                    ShogiGameRecord(
                        black_player=BLACK_PLAYER,
                        white_player=WHITE_PLAYER,
                        moves=("7g7f", "3c3d"),
                        winner="white",
                    )
                ],
            )

            examples = load_shogi_move_choice_examples_from_game_records_jsonl(path)

        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])
        self.assertEqual([example.value_target for example in examples], [-1.0, 1.0])
        self.assertEqual([example.game_index for example in examples], [0, 0])
        self.assertEqual([example.ply_index for example in examples], [0, 1])


if __name__ == "__main__":
    unittest.main()
