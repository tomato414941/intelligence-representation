from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.future_prediction_cases import extract_future_prediction_cases
from intrep.future_prediction_cases import render_future_prediction_texts
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
                    for event_id in negative_event_ids:
                        negative_event = _event_by_id(events, str(event_id))
                        self.assertTrue(str(event_id).startswith(f"env_typed_{split}_"))
                        self.assertEqual(negative_event.role, event.role)
                        self.assertEqual(negative_event.modality, event.modality)
                        self.assertNotEqual(negative_event.content, event.content)

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

    def test_generated_same_history_different_action_pairs_are_explicit_hard_negatives(self) -> None:
        _, eval_events = generated_environment_train_eval_events(
            "same_history_different_action",
            train_size=4,
            eval_size=4,
            seed=3,
        )

        self.assertEqual(len(eval_events), 12)
        self.assertEqual({event.metadata["condition"] for event in eval_events}, {"same_history_different_action"})
        self.assertEqual({event.metadata["split"] for event in eval_events}, {"eval"})
        cases = extract_future_prediction_cases(eval_events, condition="same_history_different_action")

        self.assertEqual(len(cases), 4)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            self.assertEqual(len(case.negative_events), 1)
            negative = case.negative_events[0]
            negative_events = _events_for_episode(eval_events, negative.episode_id)
            self.assertEqual(_event_content(negative_events, EventRole.OBSERVATION), positive_observation.content)
            self.assertNotEqual(_event_content(negative_events, EventRole.ACTION), positive_action.content)
            self.assertEqual(negative.role, case.positive_event.role)
            self.assertEqual(negative.modality, case.positive_event.modality)
            self.assertNotEqual(negative.content, case.positive_event.content)
            self.assertEqual(negative.metadata["pair_id"], case.positive_event.metadata["pair_id"])

    def test_generated_same_action_different_context_pairs_are_explicit_hard_negatives(self) -> None:
        _, eval_events = generated_environment_train_eval_events(
            "same_action_different_context",
            train_size=4,
            eval_size=4,
            seed=5,
        )

        self.assertEqual(len(eval_events), 12)
        self.assertEqual({event.metadata["condition"] for event in eval_events}, {"same_action_different_context"})
        self.assertEqual({event.metadata["split"] for event in eval_events}, {"eval"})
        cases = extract_future_prediction_cases(eval_events, condition="same_action_different_context")

        self.assertEqual(len(cases), 4)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            self.assertEqual(len(case.negative_events), 1)
            negative = case.negative_events[0]
            negative_events = _events_for_episode(eval_events, negative.episode_id)
            self.assertEqual(_event_content(negative_events, EventRole.ACTION), positive_action.content)
            self.assertNotEqual(_event_content(negative_events, EventRole.OBSERVATION), positive_observation.content)
            self.assertEqual(negative.role, case.positive_event.role)
            self.assertEqual(negative.modality, case.positive_event.modality)
            self.assertNotEqual(negative.content, case.positive_event.content)
            self.assertEqual(negative.metadata["pair_id"], case.positive_event.metadata["pair_id"])

    def test_same_action_different_context_content_rendering_preserves_context_signal(self) -> None:
        _, eval_events = generated_environment_train_eval_events(
            "same_action_different_context",
            train_size=4,
            eval_size=2,
            seed=7,
        )
        case = extract_future_prediction_cases(
            eval_events,
            condition="same_action_different_context",
        )[0]

        prefix, positive, negatives = render_future_prediction_texts(case, rendering="content")

        self.assertIn(case.prefix_events[0].content, prefix)
        self.assertIn(case.prefix_events[1].content, prefix)
        self.assertEqual(positive, f"{case.positive_event.content}\n")
        self.assertEqual(negatives, (f"{case.negative_events[0].content}\n",))
        self.assertNotEqual(positive, negatives[0])
        self.assertNotIn("<EVENT", prefix + positive + negatives[0])

    def test_generated_hard_negative_train_eval_have_no_id_or_negative_id_leakage(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "same_history_different_action",
            train_size=4,
            eval_size=4,
            seed=7,
        )
        train_ids = {event.id for event in train_events}
        eval_ids = {event.id for event in eval_events}

        self.assertTrue(train_ids.isdisjoint(eval_ids))
        for event in train_events:
            self.assertTrue(set(_negative_event_ids(event)).isdisjoint(eval_ids))
        for event in eval_events:
            self.assertTrue(set(_negative_event_ids(event)).isdisjoint(train_ids))

    def test_same_history_different_action_cases_keep_observation_and_change_action(self) -> None:
        events = _future_prediction_condition_events()

        cases = extract_future_prediction_cases(
            events,
            condition="same_history_different_action",
        )

        self.assertEqual(len(cases), 2)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            for negative in case.negative_events:
                negative_events = _events_for_episode(events, negative.episode_id)
                self.assertEqual(
                    _event_content(negative_events, EventRole.OBSERVATION),
                    positive_observation.content,
                )
                self.assertNotEqual(
                    _event_content(negative_events, EventRole.ACTION),
                    positive_action.content,
                )

    def test_same_action_different_context_cases_keep_action_and_change_observation(self) -> None:
        events = _future_prediction_condition_events()

        cases = extract_future_prediction_cases(
            events,
            condition="same_action_different_context",
        )

        self.assertEqual(len(cases), 2)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            for negative in case.negative_events:
                negative_events = _events_for_episode(events, negative.episode_id)
                self.assertEqual(
                    _event_content(negative_events, EventRole.ACTION),
                    positive_action.content,
                )
                self.assertNotEqual(
                    _event_content(negative_events, EventRole.OBSERVATION),
                    positive_observation.content,
                )

    def test_same_history_different_action_typed_slice_writes_pair_metadata(self) -> None:
        train_events, eval_events = generated_environment_train_eval_events(
            "same_history_different_action",
            train_size=4,
            eval_size=2,
            seed=13,
        )

        self.assertEqual(len(train_events), 12)
        self.assertEqual(len(eval_events), 6)
        self.assertEqual(_pair_conditions(train_events), {"same_history_different_action"})
        self.assertEqual(_pair_conditions(eval_events), {"same_history_different_action"})
        self.assertTrue(_pair_ids(train_events).isdisjoint(_pair_ids(eval_events)))

        cases = extract_future_prediction_cases(
            train_events,
            condition="same_history_different_action",
        )

        self.assertEqual(len(cases), 4)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            self.assertEqual(len(case.negative_events), 1)
            negative_events = _events_for_episode(
                train_events,
                case.negative_events[0].episode_id,
            )
            self.assertEqual(
                _event_content(negative_events, EventRole.OBSERVATION),
                positive_observation.content,
            )
            self.assertNotEqual(
                _event_content(negative_events, EventRole.ACTION),
                positive_action.content,
            )
            self.assertTrue(case.positive_event.metadata.get("pair_id"))

    def test_same_action_different_context_typed_slice_is_seeded_and_sized(self) -> None:
        first_train, first_eval = generated_environment_train_eval_events(
            "same_action_different_context",
            train_size=4,
            eval_size=2,
            seed=21,
        )
        second_train, second_eval = generated_environment_train_eval_events(
            "same_action_different_context",
            train_size=4,
            eval_size=2,
            seed=21,
        )

        self.assertEqual([event.id for event in first_train], [event.id for event in second_train])
        self.assertEqual([event.id for event in first_eval], [event.id for event in second_eval])
        self.assertEqual(len(first_train), 12)
        self.assertEqual(len(first_eval), 6)
        self.assertEqual(_pair_conditions(first_train), {"same_action_different_context"})

        cases = extract_future_prediction_cases(
            first_train,
            condition="same_action_different_context",
        )

        self.assertEqual(len(cases), 4)
        for case in cases:
            positive_observation, positive_action = case.prefix_events
            self.assertEqual(len(case.negative_events), 1)
            negative_events = _events_for_episode(
                first_train,
                case.negative_events[0].episode_id,
            )
            self.assertEqual(
                _event_content(negative_events, EventRole.ACTION),
                positive_action.content,
            )
            self.assertNotEqual(
                _event_content(negative_events, EventRole.OBSERVATION),
                positive_observation.content,
            )

    def test_hard_negative_typed_slices_support_100x_exp001_sizes(self) -> None:
        for split_name in (
            "same_history_different_action",
            "same_action_different_context",
        ):
            with self.subTest(split_name=split_name):
                train_events, eval_events = generated_environment_train_eval_events(
                    split_name,
                    train_size=8000,
                    eval_size=3200,
                    seed=7,
                )

                self.assertEqual(len(train_events), 24000)
                self.assertEqual(len(eval_events), 9600)
                cases = extract_future_prediction_cases(eval_events, condition=split_name)
                self.assertEqual(len(cases), 3200)
                self.assertTrue(all(len(case.negative_events) == 1 for case in cases))

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


def _event_by_id(events: list[TypedEvent], event_id: str) -> TypedEvent:
    for event in events:
        if event.id == event_id:
            return event
    raise AssertionError(f"missing event id: {event_id}")


def _pair_conditions(events: list[TypedEvent]) -> set[object]:
    return {
        event.metadata["condition"]
        for event in events
        if event.role == EventRole.CONSEQUENCE and "pair_id" in event.metadata
    }


def _pair_ids(events: list[TypedEvent]) -> set[object]:
    return {
        event.metadata["pair_id"]
        for event in events
        if event.role == EventRole.CONSEQUENCE and "pair_id" in event.metadata
    }


def _future_prediction_condition_events() -> list[TypedEvent]:
    return [
        *_episode_events("same_history_a", "door is closed", "open door", "door opens"),
        *_episode_events("same_history_b", "door is closed", "kick door", "door dents"),
        *_episode_events("same_action_a", "lamp is off", "flip switch", "lamp turns on"),
        *_episode_events("same_action_b", "fan is off", "flip switch", "fan turns on"),
    ]


def _episode_events(
    episode_id: str,
    observation: str,
    action: str,
    consequence: str,
) -> list[TypedEvent]:
    return [
        TypedEvent(
            id=f"{episode_id}_observation",
            role=EventRole.OBSERVATION,
            modality="condition_observation",
            content=observation,
            episode_id=episode_id,
            time_index=0,
        ),
        TypedEvent(
            id=f"{episode_id}_action",
            role=EventRole.ACTION,
            modality="condition_action",
            content=action,
            episode_id=episode_id,
            time_index=1,
        ),
        TypedEvent(
            id=f"{episode_id}_consequence",
            role=EventRole.CONSEQUENCE,
            modality="condition_consequence",
            content=consequence,
            episode_id=episode_id,
            time_index=2,
        ),
    ]


def _events_for_episode(events: list[TypedEvent], episode_id: str | None) -> list[TypedEvent]:
    return [event for event in events if event.episode_id == episode_id]


def _event_content(events: list[TypedEvent], role: EventRole) -> str:
    for event in events:
        if event.role == role:
            return event.content
    raise AssertionError(f"missing event role: {role.value}")


if __name__ == "__main__":
    unittest.main()
