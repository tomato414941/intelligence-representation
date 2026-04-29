from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.future_prediction_cases import extract_future_prediction_cases, render_future_prediction_texts
from intrep.generated_environment_signal_corpus import (
    generated_environment_train_eval_events,
    write_generated_environment_signal_jsonl,
)
from intrep.signal_io import load_signals_jsonl_v2


class GeneratedEnvironmentSignalCorpusTest(unittest.TestCase):
    def test_train_eval_events_include_observation_action_consequence_streams(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "generated_held_out_object"
        )

        self.assertEqual(len(train_events), 36)
        self.assertEqual(len(eval_events), 36)
        self.assertEqual(
            [event.channel for event in train_events[:3]],
            ["observation", "action", "consequence"],
        )
        self.assertIn("鍵 at 箱", train_events[0].payload)
        self.assertIn("太郎 find 鍵", train_events[1].payload)
        self.assertIn("鍵 at 棚", train_events[2].payload)

    def test_events_produce_future_prediction_cases(self) -> None:
        train_events, _ = generated_environment_train_eval_events("generated_held_out_object")

        cases = extract_future_prediction_cases(train_events)

        self.assertTrue(cases)
        self.assertEqual(cases[0].target_channel, "consequence")
        self.assertEqual([event.channel for event in cases[0].prefix_events], ["observation", "action"])
        self.assertTrue(cases[0].negative_events)

    def test_payload_rendering_uses_payload_only(self) -> None:
        train_events, _ = generated_environment_train_eval_events("generated_held_out_object")
        case = extract_future_prediction_cases(train_events)[0]

        prefix, positive, negatives = render_future_prediction_texts(case, rendering="payload")

        self.assertIn(case.prefix_events[0].payload, prefix)
        self.assertIn(case.prefix_events[1].payload, prefix)
        self.assertEqual(positive, f"{case.positive_event.payload}\n")
        self.assertEqual(negatives[0], f"{case.negative_events[0].payload}\n")

    def test_hard_negative_signal_slices_support_100x_exp001_sizes(self) -> None:
        for split_name in ("same_history_different_action", "same_action_different_context"):
            with self.subTest(split_name=split_name):
                train_events, eval_events = generated_environment_train_eval_events(
                    split_name,
                    train_size=3200,
                    eval_size=3200,
                )

                self.assertEqual(len(train_events), 9600)
                self.assertEqual(len(eval_events), 9600)
                self.assertEqual(len(extract_future_prediction_cases(train_events)), 3200)
                self.assertEqual(len(extract_future_prediction_cases(eval_events)), 3200)

    def test_writes_signal_train_eval_jsonl_v2(self) -> None:
        with TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            eval_path = Path(directory) / "eval.jsonl"

            write_generated_environment_signal_jsonl(
                train_path,
                eval_path,
                eval_slice="generated_held_out_location",
            )

            loaded_train = load_signals_jsonl_v2(train_path)
            loaded_eval = load_signals_jsonl_v2(eval_path)

        self.assertEqual(len(loaded_train), 36)
        self.assertEqual(len(loaded_eval), 18)
        self.assertEqual(loaded_train[0].channel, "observation")
        self.assertTrue(loaded_train[0].payload)


if __name__ == "__main__":
    unittest.main()
