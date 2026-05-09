import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.worlds.shogi.engine_analysis import (
    ShogiAnalysisPosition,
    ShogiEngineAnalysis,
    load_shogi_engine_analysis_jsonl,
    shogi_analysis_positions_from_game_records,
    write_shogi_engine_analysis_jsonl,
)
from intrep.worlds.shogi.game_record import ShogiActorSpec, ShogiGameRecord, shogi_game_transitions_from_usi_moves


class ShogiEngineAnalysisTest(unittest.TestCase):
    def test_round_trips_shogi_engine_analysis_jsonl(self) -> None:
        board = shogi.Board()
        record = ShogiEngineAnalysis(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            engine=ShogiActorSpec(
                kind="usi_engine",
                name="yaneuraou",
                settings={"go_command": "go nodes 10"},
            ),
            usi_info_lines=("info depth 1 nodes 10 score cp 42 pv 7g7f",),
            created_at="2026-05-09T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine-analysis.jsonl"
            write_shogi_engine_analysis_jsonl(path, [record])

            records = load_shogi_engine_analysis_jsonl(path)

        self.assertEqual(records, [record])

    def test_rejects_empty_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine-analysis.jsonl"

            with self.assertRaisesRegex(ValueError, "at least one analysis"):
                write_shogi_engine_analysis_jsonl(path, [])

    def test_extracts_unique_analysis_positions_from_game_records(self) -> None:
        first_record = _record(("7g7f", "3c3d"))
        second_record = _record(("2g2f", "8c8d"))

        positions = shogi_analysis_positions_from_game_records([first_record, second_record])

        self.assertEqual(
            positions,
            [
                ShogiAnalysisPosition(
                    position_sfen=first_record.transitions[0].position_sfen,
                    legal_moves=first_record.transitions[0].legal_moves,
                ),
                ShogiAnalysisPosition(
                    position_sfen=first_record.transitions[1].position_sfen,
                    legal_moves=first_record.transitions[1].legal_moves,
                ),
                ShogiAnalysisPosition(
                    position_sfen=second_record.transitions[1].position_sfen,
                    legal_moves=second_record.transitions[1].legal_moves,
                ),
            ],
        )


def _record(moves: tuple[str, ...]) -> ShogiGameRecord:
    actor = ShogiActorSpec(kind="test", name="actor", settings={})
    return ShogiGameRecord(
        black_actor=actor,
        white_actor=actor,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves),
    )


if __name__ == "__main__":
    unittest.main()
