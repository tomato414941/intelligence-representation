import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.worlds.shogi.engine_analysis import (
    ShogiEngineAnalysis,
    load_shogi_engine_analysis_jsonl,
    write_shogi_engine_analysis_jsonl,
)
from intrep.worlds.shogi.game_record import ShogiActorSpec


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


if __name__ == "__main__":
    unittest.main()
