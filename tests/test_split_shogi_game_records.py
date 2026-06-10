import tempfile
import unittest
from pathlib import Path

from intrep.domains.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.experience.shogi.game_split import split_shogi_game_records_jsonl


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})
USI_ENGINE_ACTOR = ShogiActorSpec(kind="usi_engine", name="yaneuraou", settings={})


def _record(
    moves: tuple[str, ...],
    winner: str | None,
    *,
    black_actor: ShogiActorSpec = BLACK_ACTOR,
    white_actor: ShogiActorSpec = WHITE_ACTOR,
) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=black_actor,
        white_actor=white_actor,
        winner=winner,
    )


def _actor_pair_counts(records: tuple[ShogiGameRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.black_actor.kind}:{record.white_actor.kind}"
        counts[key] = counts.get(key, 0) + 1
    return counts


class SplitShogiGameRecordsTest(unittest.TestCase):
    def test_splits_at_game_boundaries_by_actor_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            records = [
                _record(("7g7f",), "black", black_actor=BLACK_ACTOR, white_actor=USI_ENGINE_ACTOR),
                _record(("2g2f",), "white", black_actor=BLACK_ACTOR, white_actor=USI_ENGINE_ACTOR),
                _record(("5g5f",), "black", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
                _record(("6g6f",), "white", black_actor=USI_ENGINE_ACTOR, white_actor=USI_ENGINE_ACTOR),
            ]
            write_shogi_game_records_jsonl(games_path, records)

            train_count, eval_count = split_shogi_game_records_jsonl(
                games_jsonl=games_path,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                eval_ratio=0.25,
                seed=11,
            )

            train_records = load_shogi_game_records_jsonl(train_path)
            eval_records = load_shogi_game_records_jsonl(eval_path)
            self.assertEqual(train_count, 2)
            self.assertEqual(eval_count, 2)
            self.assertEqual(_actor_pair_counts(train_records), {"checkpoint:usi_engine": 1, "usi_engine:usi_engine": 1})
            self.assertEqual(_actor_pair_counts(eval_records), {"checkpoint:usi_engine": 1, "usi_engine:usi_engine": 1})

    def test_split_is_reproducible_for_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            records = [
                _record((move,), "black")
                for move in ("7g7f", "2g2f", "5g5f", "6g6f", "1g1f", "9g9f")
            ]
            write_shogi_game_records_jsonl(games_path, records)

            split_shogi_game_records_jsonl(
                games_jsonl=games_path,
                train_jsonl=root / "train-a.jsonl",
                eval_jsonl=root / "eval-a.jsonl",
                eval_ratio=0.33,
                seed=5,
            )
            split_shogi_game_records_jsonl(
                games_jsonl=games_path,
                train_jsonl=root / "train-b.jsonl",
                eval_jsonl=root / "eval-b.jsonl",
                eval_ratio=0.33,
                seed=5,
            )

            self.assertEqual(
                load_shogi_game_records_jsonl(root / "train-a.jsonl"),
                load_shogi_game_records_jsonl(root / "train-b.jsonl"),
            )
            self.assertEqual(
                load_shogi_game_records_jsonl(root / "eval-a.jsonl"),
                load_shogi_game_records_jsonl(root / "eval-b.jsonl"),
            )

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
