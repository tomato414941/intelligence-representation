import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )


class SplitShogiGameRecordsTest(unittest.TestCase):
    def test_splits_at_game_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            records = [
                _record(("7g7f",), "black"),
                _record(("2g2f",), "white"),
                _record(("2g2f",), "black"),
                _record(("7g7f",), "white"),
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
                    _record(("7g7f",), "black"),
                    _record(("2g2f",), "white"),
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
