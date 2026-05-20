import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import shogi

from intrep.problems.shogi_policy_value.data import (
    load_shogi_move_policy_value_examples_from_game_records_jsonl,
    shogi_engine_analysis_by_position,
    shogi_policy_targets_from_engine_analysis,
    shogi_policy_targets_from_game_record,
    shogi_return_targets_from_game_record,
    shogi_score_targets_from_engine_analysis,
    shogi_score_targets_from_game_record,
)
from intrep.problems.shogi_policy_value.examples import ShogiPolicyValueDataset
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiMoveRecord,
    shogi_game_record_from_usi_moves,
    shogi_game_record_to_json,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.game_trace import trace_shogi_game_record


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={"checkpoint": "black.pt"})
WHITE_ACTOR = ShogiActorSpec(kind="usi_engine", name="white-engine", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        winner=winner,
    )


class ShogiPolicyValueDataTest(unittest.TestCase):
    def test_builds_return_targets_from_game_outcome(self) -> None:
        record = _record(("7g7f", "3c3d"), "black")

        self.assertEqual(shogi_return_targets_from_game_record(record), (1.0, -1.0))

    def test_return_targets_are_unknown_without_outcome(self) -> None:
        record = _record(("7g7f", "3c3d"), None)

        self.assertEqual(shogi_return_targets_from_game_record(record), (None, None))

    def test_max_plies_draw_has_unknown_winner_value_targets(self) -> None:
        record = replace(_record(("7g7f", "3c3d"), None), end_reason="max_plies")

        self.assertEqual(shogi_return_targets_from_game_record(record), (None, None))

    def test_loads_move_choice_examples_from_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [_record(("7g7f", "3c3d"), "white")])

            examples = load_shogi_move_policy_value_examples_from_game_records_jsonl(
                path,
                policy_target_construction="chosen_move",
                value_target_construction="winner",
            )

        self.assertEqual([example.chosen_move for example in examples], ["7g7f", "3c3d"])
        self.assertEqual([example.value_target for example in examples], [-1.0, 1.0])
        self.assertEqual([example.game_index for example in examples], [0, 0])
        self.assertEqual([example.ply_index for example in examples], [0, 1])

    def test_loads_move_choice_examples_from_game_record_jsonl_text(self) -> None:
        record = _record(("7g7f", "3c3d"), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(action_usi="7g7f", decision_usi_info_lines=("info depth 4 score cp 100 multipv 1 pv 7g7f",)),
                record.moves[1],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            records = load_shogi_move_policy_value_examples_from_game_records_jsonl(
                path,
                policy_target_construction="decision_usi_multipv",
                value_target_construction="winner",
            )

        self.assertEqual([record.chosen_move for record in records], ["7g7f", "3c3d"])
        self.assertEqual(records[0].policy_targets, {"7g7f": 1.0})
        self.assertEqual(records[0].policy_target_source, "decision_usi_multipv")
        self.assertEqual(records[1].policy_target_source, "decision_usi_multipv")
        self.assertEqual([record.value_target for record in records], [1.0, -1.0])
        self.assertEqual([record.game_index for record in records], [0, 0])
        self.assertEqual([record.ply_index for record in records], [0, 1])

    def test_builds_score_targets_from_best_usi_score(self) -> None:
        record = _record(("7g7f", "3c3d"), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(action_usi="7g7f", decision_usi_info_lines=("info depth 4 score cp 300 multipv 1 pv 7g7f",)),
                ShogiMoveRecord(action_usi="3c3d", decision_usi_info_lines=("info depth 4 score mate -3 multipv 1 pv 3c3d",)),
            ),
        )

        targets = shogi_score_targets_from_game_record(record, score_cp_scale=300.0)

        self.assertAlmostEqual(targets[0] or 0.0, 0.761594, places=5)
        self.assertEqual(targets[1], -1.0)

    def test_builds_policy_targets_from_multipv_usi_scores(self) -> None:
        record = _record(("7g7f",), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(
                    action_usi="7g7f",
                    decision_usi_info_lines=(
                        "info multipv 1 score cp 100 pv 7g7f",
                        "info multipv 2 score cp 0 pv 2g2f",
                    ),
                ),
            ),
        )

        targets = shogi_policy_targets_from_game_record(record, source="decision_usi_multipv")[0]

        self.assertIsNotNone(targets)
        self.assertGreater(targets["7g7f"], targets["2g2f"])
        self.assertAlmostEqual(sum(targets.values()), 1.0)

    def test_builds_policy_targets_from_mcts_visit_counts(self) -> None:
        record = _record(("7g7f",), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(
                    action_usi="7g7f",
                    decision_telemetry=ShogiDecisionTelemetry(
                        search_evidence={
                            "mcts_root_child_visit_counts": {
                                "7g7f": 6,
                                "2g2f": 2,
                            }
                        },
                    ),
                ),
            ),
        )

        targets = shogi_policy_targets_from_game_record(record, source="mcts_visit_counts")[0]

        self.assertEqual(targets, {"7g7f": 0.75, "2g2f": 0.25})

    def test_loads_policy_value_examples_with_mcts_visit_target_source(self) -> None:
        record = _record(("7g7f",), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(
                    action_usi="7g7f",
                    decision_telemetry=ShogiDecisionTelemetry(
                        search_evidence={"mcts_root_child_visit_counts": {"7g7f": 6, "2g2f": 2}},
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [record])

            examples = load_shogi_move_policy_value_examples_from_game_records_jsonl(
                path,
                policy_target_construction="mcts_visit_counts",
                value_target_construction="winner",
            )

        self.assertEqual(examples[0].policy_targets, {"7g7f": 0.75, "2g2f": 0.25})
        self.assertEqual(examples[0].policy_target_source, "mcts_visit_counts")
        self.assertEqual(examples[0].value_target_source, "winner")

    def test_mcts_visit_targets_tensorize_against_candidate_order(self) -> None:
        record = _record(("7g7f",), "black")
        record = replace(
            record,
            moves=(
                ShogiMoveRecord(
                    action_usi="7g7f",
                    decision_telemetry=ShogiDecisionTelemetry(
                        search_evidence={"mcts_root_child_visit_counts": {"7g7f": 6, "2g2f": 2}},
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            write_shogi_game_records_jsonl(path, [record])
            examples = load_shogi_move_policy_value_examples_from_game_records_jsonl(
                path,
                policy_target_construction="mcts_visit_counts",
                value_target_construction="winner",
            )

        sample = ShogiPolicyValueDataset(examples)[0]

        legal_moves = examples[0].legal_moves
        self.assertEqual(int(sample.label.item()), legal_moves.index("7g7f"))
        self.assertEqual(int(sample.candidate_move_features.shape[0]), len(legal_moves))
        self.assertEqual(float(sample.policy_targets[legal_moves.index("7g7f")].item()), 0.75)
        self.assertEqual(float(sample.policy_targets[legal_moves.index("2g2f")].item()), 0.25)

    def test_builds_policy_and_score_targets_from_engine_analysis(self) -> None:
        record = _record(("7g7f",), "black")
        transition = trace_shogi_game_record(record).transitions[0]
        analysis = ShogiEngineAnalysis(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            engine=WHITE_ACTOR,
            usi_info_lines=(
                "info multipv 1 score cp 300 pv 7g7f",
                "info multipv 2 score cp 0 pv 2g2f",
            ),
        )

        analyses = shogi_engine_analysis_by_position([analysis])
        trace = trace_shogi_game_record(record)
        policy_targets = shogi_policy_targets_from_engine_analysis(analyses, trace)[0]
        score_targets = shogi_score_targets_from_engine_analysis(analyses, trace, score_cp_scale=300.0)

        self.assertIsNotNone(policy_targets)
        self.assertGreater(policy_targets["7g7f"], policy_targets["2g2f"])
        self.assertAlmostEqual(sum(policy_targets.values()), 1.0)
        self.assertAlmostEqual(score_targets[0] or 0.0, 0.761594, places=5)

    def test_missing_engine_analysis_yields_unknown_targets(self) -> None:
        record = _record(("7g7f",), "black")

        trace = trace_shogi_game_record(record)
        self.assertEqual(shogi_policy_targets_from_engine_analysis({}, trace), (None,))
        self.assertEqual(shogi_score_targets_from_engine_analysis({}, trace), (None,))

    def test_rejects_duplicate_engine_analysis_positions(self) -> None:
        record = _record(("7g7f",), "black")
        transition = trace_shogi_game_record(record).transitions[0]
        analysis = ShogiEngineAnalysis(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            engine=WHITE_ACTOR,
            usi_info_lines=(),
        )

        with self.assertRaisesRegex(ValueError, "duplicate shogi engine analysis"):
            shogi_engine_analysis_by_position([analysis, analysis])


if __name__ == "__main__":
    unittest.main()
