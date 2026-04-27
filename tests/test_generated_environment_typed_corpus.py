from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.future_prediction_cases import extract_future_prediction_cases
from intrep.generated_environment_typed_corpus import (
    generated_environment_train_eval_events,
    generated_environment_typed_corpus_selection,
    write_generated_environment_typed_jsonl,
)
from intrep.typed_corpus import load_typed_events_jsonl_v2
from intrep.typed_events import EventRole, TypedEvent


class GeneratedEnvironmentTypedCorpusTest(unittest.TestCase):
    def test_train_eval_events_include_observation_action_consequence_streams(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "generated_held_out_object"
        )

        self.assertEqual(len(train_events), 36)
        self.assertEqual(len(eval_events), 36)
        self.assertEqual(
            [event.role for event in train_events[:3]],
            [EventRole.OBSERVATION, EventRole.ACTION, EventRole.CONSEQUENCE],
        )
        self.assertEqual([event.time_index for event in train_events[:3]], [0, 1, 2])
        self.assertTrue(all(event.episode_id for event in train_events))
        self.assertIn("鍵 at 箱", train_events[0].content)
        self.assertIn("太郎 find 鍵", train_events[1].content)
        self.assertIn("鍵 at 棚", train_events[2].content)

    def test_selection_records_eval_slice(self) -> None:
        selection = generated_environment_typed_corpus_selection(
            "generated_held_out_container"
        )

        self.assertEqual(selection.eval_label, "generated_held_out_container")
        self.assertEqual(len(selection.train_events), 36)
        self.assertEqual(len(selection.eval_events), 36)
        self.assertEqual(
            {event.metadata["split"] for event in selection.train_events},
            {"train"},
        )
        self.assertEqual(
            {event.metadata["eval_slice"] for event in selection.eval_events},
            {"generated_held_out_container"},
        )

    def test_consequence_events_have_explicit_same_split_negative_event_ids(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "generated_held_out_object"
        )

        for events, split in ((train_events, "train"), (eval_events, "eval")):
            with self.subTest(split=split):
                event_ids = {event.id for event in events}
                consequence_events = [
                    event for event in events if event.role == EventRole.CONSEQUENCE
                ]

                self.assertTrue(consequence_events)
                self.assertTrue(
                    all(event.metadata.get("negative_event_ids") for event in consequence_events)
                )
                for event in consequence_events:
                    negative_event_ids = event.metadata["negative_event_ids"]
                    self.assertIsInstance(negative_event_ids, list)
                    self.assertTrue(set(negative_event_ids).issubset(event_ids))
                    self.assertTrue(
                        all(
                            str(event_id).startswith(f"env_typed_{split}_")
                            for event_id in negative_event_ids
                        )
                    )

    def test_train_eval_event_ids_do_not_leak_even_for_seen_slice(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "generated_seen"
        )

        train_ids = {event.id for event in train_events}
        eval_ids = {event.id for event in eval_events}

        self.assertTrue(train_ids)
        self.assertTrue(eval_ids)
        self.assertTrue(train_ids.isdisjoint(eval_ids))
        self.assertEqual({event.metadata["split"] for event in train_events}, {"train"})
        self.assertEqual({event.metadata["split"] for event in eval_events}, {"eval"})

    def test_split_metadata_negative_ids_do_not_cross_train_eval(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "generated_held_out_location"
        )
        train_ids = {event.id for event in train_events}
        eval_ids = {event.id for event in eval_events}

        for event in train_events:
            negative_event_ids = set(_negative_event_ids(event))
            self.assertTrue(negative_event_ids.isdisjoint(eval_ids))
        for event in eval_events:
            negative_event_ids = set(_negative_event_ids(event))
            self.assertTrue(negative_event_ids.isdisjoint(train_ids))

    def test_events_produce_future_prediction_cases_from_explicit_negatives(self) -> None:
        _, eval_events = generated_environment_train_eval_events("generated_strict_noisy")

        cases = extract_future_prediction_cases(eval_events)

        self.assertEqual(len(cases), 4)
        self.assertTrue(all(case.negative_events for case in cases))
        self.assertTrue(
            all(
                negative.id in case.positive_event.metadata["negative_event_ids"]
                for case in cases
                for negative in case.negative_events
            )
        )

    def test_writes_typed_train_eval_jsonl_v2(self) -> None:
        with TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.typed.jsonl"
            eval_path = Path(directory) / "eval.typed.jsonl"

            selection = write_generated_environment_typed_jsonl(
                train_path,
                eval_path,
                eval_slice="generated_strict_noisy",
            )

            loaded_train = load_typed_events_jsonl_v2(train_path)
            loaded_eval = load_typed_events_jsonl_v2(eval_path)

        self.assertEqual(len(loaded_train), len(selection.train_events))
        self.assertEqual(len(loaded_eval), len(selection.eval_events))
        self.assertEqual({event.metadata["split"] for event in loaded_train}, {"train"})
        self.assertEqual({event.metadata["split"] for event in loaded_eval}, {"eval"})


def _negative_event_ids(event: TypedEvent) -> list[str]:
    value = event.metadata.get("negative_event_ids", [])
    return [str(event_id) for event_id in value]


if __name__ == "__main__":
    unittest.main()
