import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.worlds.shogi.position_index import (
    ShogiPositionIndex,
    ShogiPositionIndexEntry,
    ShogiPositionIndexRecord,
    load_shogi_position_index_jsonl,
    write_shogi_position_index_jsonl,
)


class ShogiPositionIndexTest(unittest.TestCase):
    def test_round_trips_shogi_position_index_jsonl(self) -> None:
        record = ShogiPositionIndexRecord(
            position_sfen=shogi.Board().sfen(),
            entries=(
                ShogiPositionIndexEntry(
                    kind="best_move",
                    source="yaneuraou",
                    created_at="2026-05-09T00:00:00Z",
                    settings={"go_command": "go nodes 1"},
                    data={"move_usi": "7g7f"},
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "position-index.jsonl"

            write_shogi_position_index_jsonl(path, [record])
            index = load_shogi_position_index_jsonl(path)

        loaded = index.get(record.position_sfen)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.entries[0].kind, "best_move")
        self.assertEqual(loaded.entries[0].data["move_usi"], "7g7f")
        self.assertEqual(loaded.entries[0].settings["go_command"], "go nodes 1")

    def test_filters_entries_by_kind_and_source(self) -> None:
        position_sfen = shogi.Board().sfen()
        index = ShogiPositionIndex(
            [
                ShogiPositionIndexRecord(
                    position_sfen=position_sfen,
                    entries=(
                        ShogiPositionIndexEntry(kind="best_move", source="yaneuraou", data={"move_usi": "7g7f"}),
                        ShogiPositionIndexEntry(kind="value", source="self_mcts", data={"value": 0.25}),
                    ),
                )
            ]
        )

        entries = index.entries(position_sfen, kind="best_move")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].source, "yaneuraou")
        self.assertEqual(index.entries(position_sfen, source="missing"), ())

    def test_rejects_duplicate_position_records(self) -> None:
        position_sfen = shogi.Board().sfen()
        record = ShogiPositionIndexRecord(
            position_sfen=position_sfen,
            entries=(ShogiPositionIndexEntry(kind="best_move", source="yaneuraou", data={"move_usi": "7g7f"}),),
        )

        with self.assertRaises(ValueError):
            ShogiPositionIndex([record, record])

    def test_rejects_empty_entries(self) -> None:
        with self.assertRaises(ValueError):
            ShogiPositionIndexRecord(position_sfen=shogi.Board().sfen(), entries=())


if __name__ == "__main__":
    unittest.main()
