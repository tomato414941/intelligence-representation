import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO

import shogi

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    ShogiTransitionRecord,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.info_stats import inspect_shogi_usi_info_jsonl
from intrep.worlds.shogi.inspect_usi_info import main


BLACK_ACTOR = ShogiActorSpec(kind="baseline", name="black", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="yaneuraou", name="white", settings={"go_command": "go nodes 10"})


class ShogiUsiInfoStatsTest(unittest.TestCase):
    def test_inspects_raw_decision_usi_info_lines(self) -> None:
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
        self.assertEqual(stats["best_score_ply_count"], 1)
        self.assertEqual(stats["best_score_cp_line_count"], 1)
        self.assertEqual(stats["best_score_mate_line_count"], 0)
        self.assertEqual(stats["best_score_cp_min"], 23)
        self.assertEqual(stats["best_score_cp_max"], 23)
        self.assertEqual(stats["best_score_cp_mean"], 23.0)
        self.assertIsNone(stats["best_score_mate_min"])
        self.assertIsNone(stats["best_score_mate_max"])
        self.assertEqual(stats["depth_line_count"], 2)
        self.assertEqual(stats["nodes_line_count"], 2)
        self.assertEqual(stats["pv_line_count"], 2)
        self.assertEqual(stats["multipv_line_count"], 1)
        self.assertEqual(stats["action_pv_match_count"], 1)
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
    transitions = shogi_game_transitions_from_usi_moves(("7g7f", "3c3d"))
    second = transitions[1]
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=(
            transitions[0],
            ShogiTransitionRecord(
                ply=second.ply,
                side=second.side,
                position_sfen=second.position_sfen,
                legal_moves=second.legal_moves,
                action_usi=second.action_usi,
                next_position_sfen=second.next_position_sfen,
                reward=second.reward,
                done=second.done,
                decision_usi_info_lines=(
                    "info depth 4 nodes 100 score cp 23 pv 3c3d 2g2f",
                    "info multipv 2 depth 3 nodes 80 score mate -5 pv 8c8d",
                ),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
