import json
import tempfile
import unittest
from pathlib import Path

import shogi

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
    load_shogi_game_records_jsonl,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={"checkpoint": "black.pt"})
WHITE_ACTOR = ShogiActorSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None, end_reason: str | None = None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
        end_reason=end_reason,
    )


class ShogiGameRecordTest(unittest.TestCase):
    def test_round_trips_shogi_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            expected = [
                _record(("7g7f", "3c3d"), "white", "resign"),
                _record(("2g2f",), None, "max_plies"),
            ]

            write_shogi_game_records_jsonl(path, expected)
            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(records, expected)

    def test_loads_shogi_game_record_jsonl(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        board.push_usi("2g2f")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "black_actor": {
                            "kind": "checkpoint",
                            "name": "direct",
                            "settings": {"checkpoint": "model.pt"},
                        },
                        "white_actor": {
                            "kind": "yaneuraou",
                            "name": "yaneuraou",
                            "settings": {"go_command": "go nodes 1"},
                        },
                        "initial_position_sfen": shogi.Board().sfen(),
                        "transitions": [
                            {
                                "ply": 0,
                                "side": "black",
                                "position_sfen": shogi.Board().sfen(),
                                "legal_moves": list(legal_moves),
                                "action_usi": "2g2f",
                                "next_position_sfen": board.sfen(),
                                "reward": 0.0,
                                "done": True,
                                "decision_usi_info_lines": ["info depth 1 nodes 1 pv 2g2f"],
                            }
                        ],
                        "end_reason": "max_plies",
                        "winner": None,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(
            records,
            [
                ShogiGameRecord(
                    black_actor=ShogiActorSpec(kind="checkpoint", name="direct", settings={"checkpoint": "model.pt"}),
                    white_actor=ShogiActorSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 1"}),
                    initial_position_sfen=shogi.Board().sfen(),
                    transitions=(
                        ShogiTransitionRecord(
                            ply=0,
                            side="black",
                            position_sfen=shogi.Board().sfen(),
                            legal_moves=legal_moves,
                            action_usi="2g2f",
                            next_position_sfen=board.sfen(),
                            reward=0.0,
                            done=True,
                            decision_usi_info_lines=("info depth 1 nodes 1 pv 2g2f",),
                        ),
                    ),
                    winner=None,
                    end_reason="max_plies",
                )
            ],
        )

    def test_loads_transition_without_teacher_targets(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        board.push_usi("2g2f")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "black_actor": {"kind": "checkpoint", "name": "direct", "settings": {}},
                        "white_actor": {"kind": "yaneuraou", "name": "yaneuraou", "settings": {}},
                        "initial_position_sfen": shogi.Board().sfen(),
                        "transitions": [
                            {
                                "ply": 0,
                                "side": "black",
                                "position_sfen": shogi.Board().sfen(),
                                "legal_moves": list(legal_moves),
                                "action_usi": "2g2f",
                                "next_position_sfen": board.sfen(),
                                "reward": 0.0,
                                "done": True,
                                "decision_usi_info_lines": [],
                            }
                        ],
                        "end_reason": "max_plies",
                        "winner": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(records[0].transitions[0].action_usi, "2g2f")

    def test_round_trips_decision_telemetry(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        board.push_usi("2g2f")
        record = ShogiGameRecord(
            black_actor=ShogiActorSpec(kind="checkpoint", name="direct", settings={}),
            white_actor=ShogiActorSpec(kind="yaneuraou", name="yaneuraou", settings={}),
            initial_position_sfen=shogi.Board().sfen(),
            transitions=(
                ShogiTransitionRecord(
                    ply=0,
                    side="black",
                    position_sfen=shogi.Board().sfen(),
                    legal_moves=legal_moves,
                    action_usi="2g2f",
                    next_position_sfen=board.sfen(),
                    reward=0.0,
                    done=True,
                    decision_telemetry=ShogiDecisionTelemetry(
                        move_performance={"request_wall_time_sec": 0.4},
                        batch_performance={"position_count": 4},
                    ),
                ),
            ),
            winner=None,
            end_reason="max_plies",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [record])
            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(records, [record])

    def test_loads_legacy_performance_info_lines_as_decision_telemetry(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        board.push_usi("2g2f")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "black_actor": {"kind": "checkpoint", "name": "direct", "settings": {}},
                        "white_actor": {"kind": "yaneuraou", "name": "yaneuraou", "settings": {}},
                        "initial_position_sfen": shogi.Board().sfen(),
                        "transitions": [
                            {
                                "ply": 0,
                                "side": "black",
                                "position_sfen": shogi.Board().sfen(),
                                "legal_moves": list(legal_moves),
                                "action_usi": "2g2f",
                                "next_position_sfen": board.sfen(),
                                "reward": 0.0,
                                "done": True,
                                "decision_usi_info_lines": [
                                    "info depth 1 nodes 1 pv 2g2f",
                                    'info string intrep_performance {"request_wall_time_sec": 0.4}',
                                    'info string intrep_batch_performance {"position_count": 4}',
                                ],
                            }
                        ],
                        "end_reason": "max_plies",
                        "winner": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_shogi_game_records_jsonl(path)

        transition = records[0].transitions[0]
        self.assertEqual(transition.decision_usi_info_lines, ("info depth 1 nodes 1 pv 2g2f",))
        self.assertIsNotNone(transition.decision_telemetry)
        assert transition.decision_telemetry is not None
        self.assertEqual(transition.decision_telemetry.move_performance, {"request_wall_time_sec": 0.4})
        self.assertEqual(transition.decision_telemetry.batch_performance, {"position_count": 4})


if __name__ == "__main__":
    unittest.main()
