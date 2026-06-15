from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from intrep.problems.shogi_policy_value.data_selection import (
    load_shogi_policy_value_data_selection,
    load_shogi_policy_value_data_selection_examples,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiMovePolicyValueExample,
    load_shogi_move_policy_value_examples_jsonl,
    write_shogi_move_policy_value_examples_jsonl,
)
from intrep.problems.shogi_policy_value.split_data_selection import (
    split_shogi_policy_value_data_selection,
)
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)


ACTOR = ShogiActorSpec(kind="test", name="test", settings={})


def _example(game_index: int, ply_index: int, move: str) -> ShogiMovePolicyValueExample:
    return ShogiMovePolicyValueExample(
        position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        legal_moves=("7g7f", "2g2f", "5g5f"),
        chosen_move=move,
        value_target=1.0,
        game_index=game_index,
        ply_index=ply_index,
    )


class ShogiSplitDataSelectionTest(unittest.TestCase):
    def test_splits_all_sources_into_new_data_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_examples_path = root / "qhapaq-train.jsonl"
            eval_examples_path = root / "qhapaq-eval.jsonl"
            games_path = root / "self-play.games.jsonl"
            data_selection_path = root / "data-selection.json"
            output_dir = root / "split-selection"
            write_shogi_move_policy_value_examples_jsonl(
                train_examples_path,
                [
                    _example(1, 0, "7g7f"),
                    _example(1, 1, "2g2f"),
                    _example(2, 0, "5g5f"),
                    _example(2, 1, "7g7f"),
                ],
            )
            write_shogi_move_policy_value_examples_jsonl(
                eval_examples_path,
                [
                    _example(3, 0, "7g7f"),
                    _example(3, 1, "2g2f"),
                    _example(4, 0, "5g5f"),
                    _example(4, 1, "7g7f"),
                ],
            )
            write_shogi_game_records_jsonl(
                games_path,
                [
                    shogi_game_record_from_usi_moves(
                        ("7g7f", "3c3d"),
                        black_actor=ACTOR,
                        white_actor=ACTOR,
                        winner="black",
                        end_reason="game_over",
                    ),
                    shogi_game_record_from_usi_moves(
                        ("2g2f", "8c8d"),
                        black_actor=ACTOR,
                        white_actor=ACTOR,
                        winner="white",
                        end_reason="game_over",
                    ),
                ],
            )
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "source-selection",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [
                            {"kind": "shogi_policy_value_examples_jsonl", "path": str(train_examples_path)},
                            {"kind": "game_records_jsonl", "path": str(games_path)},
                        ],
                        "eval_sources": [
                            {"kind": "shogi_policy_value_examples_jsonl", "path": str(eval_examples_path)},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = split_shogi_policy_value_data_selection(
                data_selection_path=data_selection_path,
                output_dir=output_dir,
                name="split-selection",
                eval_ratio=0.5,
                seed=11,
            )
            output_selection = load_shogi_policy_value_data_selection(result.data_selection_path)
            train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(output_selection)
            self.assertTrue(result.manifest_path.exists())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.train_count, 6)
        self.assertEqual(result.eval_count, 6)
        self.assertEqual(output_selection.name, "split-selection")
        self.assertEqual(len(output_selection.train_sources), 3)
        self.assertEqual(len(output_selection.eval_sources), 3)
        self.assertEqual(len(train_examples), 6)
        self.assertEqual(len(eval_examples), 6)
        self.assertEqual(manifest["schema_version"], "intrep.shogi_policy_value_split_data_selection.v1")
        self.assertEqual(manifest["train_count"], 6)
        self.assertEqual(manifest["eval_count"], 6)
        self.assertEqual(len(manifest["sources"]), 3)

    def test_splits_examples_by_game_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples_path = root / "examples.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            write_shogi_move_policy_value_examples_jsonl(
                examples_path,
                [
                    _example(1, 0, "7g7f"),
                    _example(1, 1, "2g2f"),
                    _example(2, 0, "5g5f"),
                    _example(2, 1, "7g7f"),
                ],
            )

            from intrep.problems.shogi_policy_value.split_data_selection import _split_examples_jsonl

            train_count, eval_count = _split_examples_jsonl(
                examples_jsonl=examples_path,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                eval_ratio=0.5,
                seed=7,
            )
            train_game_indices = {
                example.game_index for example in load_shogi_move_policy_value_examples_jsonl(train_path)
            }
            eval_game_indices = {
                example.game_index for example in load_shogi_move_policy_value_examples_jsonl(eval_path)
            }

        self.assertEqual(train_count, 2)
        self.assertEqual(eval_count, 2)
        self.assertEqual(len(train_game_indices), 1)
        self.assertEqual(len(eval_game_indices), 1)
        self.assertFalse(train_game_indices & eval_game_indices)


if __name__ == "__main__":
    unittest.main()
