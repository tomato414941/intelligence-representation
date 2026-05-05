import tempfile
import unittest
from pathlib import Path

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord, shogi_game_ply_records_from_usi_moves, load_shogi_game_records_jsonl, write_shogi_game_records_jsonl
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={})
WHITE_PLAYER = PlayerSpec(kind="checkpoint", name="white-model", settings={})


class SplitShogiGameRecordsTest(unittest.TestCase):
    def test_splits_at_game_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            records = [
                ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("7g7f",)), "black"),
                ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("3c3d",)), "white"),
                ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("2g2f",)), "black"),
                ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("8c8d",)), "white"),
            ]
            write_shogi_game_records_jsonl(games_path, records)

            train_count, eval_count = split_shogi_game_records_jsonl(
                games_jsonl=games_path,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                eval_ratio=0.25,
            )

            self.assertEqual(train_count, 3)
            self.assertEqual(eval_count, 1)
            self.assertEqual(load_shogi_game_records_jsonl(train_path), records[:3])
            self.assertEqual(load_shogi_game_records_jsonl(eval_path), records[3:])

    def test_rejects_invalid_eval_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            write_shogi_game_records_jsonl(
                games_path,
                [
                    ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("7g7f",)), "black"),
                    ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(("3c3d",)), "white"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "eval-ratio"):
                split_shogi_game_records_jsonl(
                    games_jsonl=games_path,
                    train_jsonl=root / "train.jsonl",
                    eval_jsonl=root / "eval.jsonl",
                    eval_ratio=0.0,
                )


if __name__ == "__main__":
    unittest.main()
