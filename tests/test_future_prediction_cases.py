import unittest

from intrep.future_prediction_cases import (
    extract_future_prediction_cases,
    render_future_prediction_continuations,
    render_future_prediction_prefix,
)
from intrep.typed_events import EventRole, TypedEvent


class FuturePredictionCasesTest(unittest.TestCase):
    def test_extracts_consequence_cases_directly_from_typed_events(self) -> None:
        events = [
            _event("ep1_obs", "observation", "grid", "A..", "ep1", 0),
            _event("ep1_action", "action", "grid_action", "right", "ep1", 1),
            _event("ep1_cons", "consequence", "grid", ".A.", "ep1", 2),
            _event("ep2_obs", "observation", "grid", "B..", "ep2", 0),
            _event("ep2_action", "action", "grid_action", "left", "ep2", 1),
            _event("ep2_cons", "consequence", "grid", "..B", "ep2", 2),
        ]

        cases = extract_future_prediction_cases(events)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].target_role, EventRole.CONSEQUENCE)
        self.assertEqual([event.id for event in cases[0].prefix_events], ["ep1_obs", "ep1_action"])
        self.assertEqual(cases[0].positive_event.id, "ep1_cons")
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep2_cons"])

    def test_extracts_tool_call_to_tool_result_cases(self) -> None:
        events = [
            _event("ep1_call", "tool_call", "search", "query key", "ep1", 0),
            _event("ep1_result", "tool_result", "search_result", "key in box", "ep1", 1),
            _event("ep2_call", "tool_call", "search", "query coin", "ep2", 0),
            _event("ep2_result", "tool_result", "search_result", "coin in drawer", "ep2", 1),
        ]

        cases = extract_future_prediction_cases(events, target_role="tool_result")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].positive_event.role, EventRole.TOOL_RESULT)
        self.assertEqual(cases[0].negative_events[0].id, "ep2_result")

    def test_extracts_prediction_to_prediction_error_cases(self) -> None:
        events = [
            _event("ep1_pred", "prediction", "text", "see key", "ep1", 0),
            _event("ep1_error", "prediction_error", "text", "correct", "ep1", 1),
            _event("ep2_pred", "prediction", "text", "see coin", "ep2", 0),
            _event("ep2_error", "prediction_error", "text", "wrong_object", "ep2", 1),
        ]

        cases = extract_future_prediction_cases(events, target_role="prediction_error")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].prefix_events[0].role, EventRole.PREDICTION)
        self.assertEqual(cases[0].positive_event.role, EventRole.PREDICTION_ERROR)

    def test_tool_call_result_prefers_matching_tool_call_id(self) -> None:
        events = [
            _event("ep1_call", "tool_call", "search", "query key", "ep1", 0, {"tool_call_id": "call-1"}),
            _event("ep1_wrong", "tool_result", "search_result", "wrong", "ep1", 1, {"tool_call_id": "call-2"}),
            _event("ep1_result", "tool_result", "search_result", "key in box", "ep1", 2, {"tool_call_id": "call-1"}),
            _event("ep2_result", "tool_result", "search_result", "coin in drawer", "ep2", 1),
        ]

        cases = extract_future_prediction_cases(events, target_role="tool_result")

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].positive_event.id, "ep1_result")

    def test_same_action_different_context_filters_negative_actions(self) -> None:
        events = [
            _event("ep1_obs", "observation", "text", "key in box", "ep1", 0),
            _event("ep1_action", "action", "text", "open", "ep1", 1),
            _event("ep1_cons", "consequence", "text", "see key", "ep1", 2),
            _event("ep2_obs", "observation", "text", "coin in box", "ep2", 0),
            _event("ep2_action", "action", "text", "open", "ep2", 1),
            _event("ep2_cons", "consequence", "text", "see coin", "ep2", 2),
            _event("ep3_obs", "observation", "text", "map in box", "ep3", 0),
            _event("ep3_action", "action", "text", "shake", "ep3", 1),
            _event("ep3_cons", "consequence", "text", "hear paper", "ep3", 2),
        ]

        cases = extract_future_prediction_cases(
            events,
            condition="same_action_different_context",
        )

        self.assertEqual(len(cases), 2)
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep2_cons"])

    def test_same_action_different_context_requires_different_history(self) -> None:
        events = [
            _event("ep1_obs", "observation", "text", "key in box", "ep1", 0),
            _event("ep1_action", "action", "text", "open", "ep1", 1),
            _event("ep1_cons", "consequence", "text", "see key", "ep1", 2),
            _event("ep2_obs", "observation", "text", "coin in box", "ep2", 0),
            _event("ep2_action", "action", "text", "open", "ep2", 1),
            _event("ep2_cons", "consequence", "text", "see coin", "ep2", 2),
            _event("ep3_obs", "observation", "text", "key in box", "ep3", 0),
            _event("ep3_action", "action", "text", "open", "ep3", 1),
            _event("ep3_cons", "consequence", "text", "see decoy", "ep3", 2),
        ]

        cases = extract_future_prediction_cases(
            events,
            condition="same_action_different_context",
        )

        self.assertEqual(len(cases), 3)
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep2_cons"])

    def test_same_history_different_action_filters_negative_history(self) -> None:
        events = [
            _event("ep1_obs", "observation", "text", "box_a key ; box_b coin", "ep1", 0),
            _event("ep1_action", "action", "text", "open box_a", "ep1", 1),
            _event("ep1_cons", "consequence", "text", "see key", "ep1", 2),
            _event("ep2_obs", "observation", "text", "box_a key ; box_b coin", "ep2", 0),
            _event("ep2_action", "action", "text", "open box_b", "ep2", 1),
            _event("ep2_cons", "consequence", "text", "see coin", "ep2", 2),
        ]

        cases = extract_future_prediction_cases(
            events,
            condition="same_history_different_action",
        )

        self.assertEqual(len(cases), 2)
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep2_cons"])

    def test_same_history_different_action_requires_different_action(self) -> None:
        events = [
            _event("ep1_obs", "observation", "text", "box_a key ; box_b coin", "ep1", 0),
            _event("ep1_action", "action", "text", "open box_a", "ep1", 1),
            _event("ep1_cons", "consequence", "text", "see key", "ep1", 2),
            _event("ep2_obs", "observation", "text", "box_a key ; box_b coin", "ep2", 0),
            _event("ep2_action", "action", "text", "open box_b", "ep2", 1),
            _event("ep2_cons", "consequence", "text", "see coin", "ep2", 2),
            _event("ep3_obs", "observation", "text", "box_a key ; box_b coin", "ep3", 0),
            _event("ep3_action", "action", "text", "open box_a", "ep3", 1),
            _event("ep3_cons", "consequence", "text", "see decoy", "ep3", 2),
        ]

        cases = extract_future_prediction_cases(
            events,
            condition="same_history_different_action",
        )

        self.assertEqual(len(cases), 3)
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep2_cons"])

    def test_explicit_negative_ids_override_candidate_filtering(self) -> None:
        events = [
            _event("ep1_obs", "observation", "text", "key in box", "ep1", 0),
            _event("ep1_action", "action", "text", "open", "ep1", 1),
            _event(
                "ep1_cons",
                "consequence",
                "text",
                "see key",
                "ep1",
                2,
                {"negative_event_ids": ["ep3_cons"]},
            ),
            _event("ep2_obs", "observation", "text", "coin in box", "ep2", 0),
            _event("ep2_action", "action", "text", "open", "ep2", 1),
            _event("ep2_cons", "consequence", "text", "see coin", "ep2", 2),
            _event("ep3_obs", "observation", "grid", "map in box", "ep3", 0),
            _event("ep3_action", "action", "grid_action", "shake", "ep3", 1),
            _event("ep3_cons", "consequence", "grid", "hear paper", "ep3", 2),
        ]

        cases = extract_future_prediction_cases(
            events,
            condition="same_action_different_context",
        )

        self.assertEqual(cases[0].positive_event.id, "ep1_cons")
        self.assertEqual([event.id for event in cases[0].negative_events], ["ep3_cons"])

    def test_renders_prefix_and_continuations_with_typed_stream(self) -> None:
        events = [
            _event("ep1_obs", "observation", "grid", "A..", "ep1", 0),
            _event("ep1_action", "action", "grid_action", "right", "ep1", 1),
            _event("ep1_cons", "consequence", "grid", ".A.", "ep1", 2),
            _event("ep2_cons", "consequence", "grid", "..B", "ep2", 2),
        ]
        case = extract_future_prediction_cases(events)[0]

        prefix = render_future_prediction_prefix(case)
        continuations = render_future_prediction_continuations(case)

        self.assertIn('role="observation"', prefix)
        self.assertIn('role="action"', prefix)
        self.assertNotIn(".A.", prefix)
        self.assertEqual(len(continuations), 2)
        self.assertIn(".A.", continuations[0])
        self.assertIn("..B", continuations[1])


def _event(
    event_id: str,
    role: str,
    modality: str,
    content: str,
    episode_id: str,
    time_index: int,
    metadata: dict[str, object] | None = None,
) -> TypedEvent:
    return TypedEvent(
        id=event_id,
        role=role,
        modality=modality,
        content=content,
        episode_id=episode_id,
        time_index=time_index,
        metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
