import tempfile
import unittest
from pathlib import Path

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord, shogi_game_ply_records_from_usi_moves, write_shogi_game_records_jsonl
from intrep.tasks.shogi_move_choice.data import load_shogi_move_choice_examples_from_game_records_jsonl


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
                        plies=shogi_game_ply_records_from_usi_moves(("7g7f", "3c3d")),
                        winner="white",
                    )
                ],
            )

            examples = load_shogi_move_choice_examples_from_game_records_jsonl(path)

        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])
        self.assertEqual([example.value_target for example in examples], [-1.0, 1.0])
        self.assertEqual([example.game_index for example in examples], [0, 0])
        self.assertEqual([example.ply_index for example in examples], [0, 1])

    def test_loads_move_choice_examples_from_arena_game_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena-games.jsonl"
            path.write_text(
                (
                    '{"black_player":{"kind":"checkpoint","name":"checkpoint-direct-black",'
                    '"settings":{"checkpoint":"black.pt","policy":"direct","simulations":null}},'
                    '"white_player":{"kind":"checkpoint","name":"checkpoint-direct-white",'
                    '"settings":{"checkpoint":"white.pt","policy":"direct","simulations":null}},'
                    '"plies":[{"side":"black","position":"position startpos","bestmove":"7g7f",'
                    '"ponder":"3c3d","usi_info_lines":["info depth 2 nodes 10 pv 7g7f 3c3d"]},'
                    '{"side":"white","position":"position startpos moves 7g7f","bestmove":"3c3d",'
                    '"ponder":null,"usi_info_lines":[]}],'
                    '"end_reason":"resign","winner":"black"}\n'
                ),
                encoding="utf-8",
            )

            records = load_shogi_move_choice_examples_from_game_records_jsonl(path)

        self.assertEqual([record.chosen_move for record in records], ["7g7f", "3c3d"])
        self.assertEqual([record.value_target for record in records], [1.0, -1.0])
        self.assertEqual([record.game_index for record in records], [0, 0])
        self.assertEqual([record.ply_index for record in records], [0, 1])


if __name__ == "__main__":
    unittest.main()
