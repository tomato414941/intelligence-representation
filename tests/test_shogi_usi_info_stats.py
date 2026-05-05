import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO

from intrep.inspect_shogi_usi_info import main
from intrep.worlds.shogi.game_record import (
    PlayerSpec,
    ShogiGamePlyRecord,
    ShogiGameRecord,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.info_stats import inspect_shogi_usi_info_jsonl


BLACK_PLAYER = PlayerSpec(kind="baseline", name="black", settings={})
WHITE_PLAYER = PlayerSpec(kind="yaneuraou", name="white", settings={"go_command": "go nodes 10"})


class ShogiUsiInfoStatsTest(unittest.TestCase):
    def test_inspects_raw_usi_info_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [_record()])

            stats = inspect_shogi_usi_info_jsonl(path).to_dict()

        self.assertEqual(stats["game_count"], 1)
        self.assertEqual(stats["ply_count"], 2)
        self.assertEqual(stats["info_ply_count"], 1)
        self.assertEqual(stats["info_line_count"], 2)
        self.assertEqual(stats["score_cp_line_count"], 1)
        self.assertEqual(stats["score_mate_line_count"], 1)
        self.assertEqual(stats["depth_line_count"], 2)
        self.assertEqual(stats["nodes_line_count"], 2)
        self.assertEqual(stats["pv_line_count"], 2)
        self.assertEqual(stats["multipv_line_count"], 1)
        self.assertEqual(stats["bestmove_pv_match_count"], 1)
        self.assertEqual(stats["multipv_counts"], {"2": 1})
        self.assertEqual(stats["depth_counts"], {"3": 1, "4": 1})
        self.assertEqual(stats["nodes_min"], 80)
        self.assertEqual(stats["nodes_max"], 100)

    def test_cli_writes_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(games_path, [_record()])

            with patch(
                "sys.argv",
                [
                    "inspect_shogi_usi_info",
                    "--games-jsonl",
                    str(games_path),
                    "--metrics-json",
                    str(metrics_path),
                ],
            ), patch("sys.stdout", new_callable=StringIO) as stdout:
                main()

            stdout_payload = json.loads(stdout.getvalue())
            file_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(stdout_payload["info_line_count"], 2)
        self.assertEqual(file_payload["info_line_count"], 2)


def _record() -> ShogiGameRecord:
    return ShogiGameRecord(
        black_player=BLACK_PLAYER,
        white_player=WHITE_PLAYER,
        plies=(
            ShogiGamePlyRecord(side="black", position="position startpos", bestmove="7g7f"),
            ShogiGamePlyRecord(
                side="white",
                position="position startpos moves 7g7f",
                bestmove="3c3d",
                usi_info_lines=(
                    "info depth 4 nodes 100 score cp 23 pv 3c3d 2g2f",
                    "info multipv 2 depth 3 nodes 80 score mate -5 pv 8c8d",
                ),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
