import unittest

from intrep.future_prediction_cases import (
    extract_future_prediction_cases,
    render_future_prediction_continuations,
    render_future_prediction_prefix,
    render_future_prediction_texts,
)
from intrep.signals import Signal


class FuturePredictionCasesTest(unittest.TestCase):
    def test_extracts_consequence_cases_from_ordered_signal_stream(self) -> None:
        events = [
            Signal(channel="observation", payload="A.."),
            Signal(channel="action", payload="right"),
            Signal(channel="consequence", payload=".A."),
            Signal(channel="observation", payload="B.."),
            Signal(channel="action", payload="left"),
            Signal(channel="consequence", payload="..B"),
        ]

        cases = extract_future_prediction_cases(events)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].target_channel, "consequence")
        self.assertEqual([event.payload for event in cases[0].prefix_events], ["A..", "right"])
        self.assertEqual(cases[0].positive_event.payload, ".A.")
        self.assertEqual([event.payload for event in cases[0].negative_events], ["..B"])

    def test_extracts_tool_call_to_tool_result_cases(self) -> None:
        events = [
            Signal(channel="tool_call", payload="query key"),
            Signal(channel="tool_result", payload="key in box"),
            Signal(channel="tool_call", payload="query coin"),
            Signal(channel="tool_result", payload="coin in drawer"),
        ]

        cases = extract_future_prediction_cases(events, target_channel="tool_result")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].positive_event.payload, "key in box")
        self.assertEqual(cases[0].negative_events[0].payload, "coin in drawer")

    def test_extracts_prediction_to_prediction_error_cases(self) -> None:
        events = [
            Signal(channel="prediction", payload="see key"),
            Signal(channel="prediction_error", payload="correct"),
            Signal(channel="prediction", payload="see coin"),
            Signal(channel="prediction_error", payload="wrong_object"),
        ]

        cases = extract_future_prediction_cases(events, target_channel="prediction_error")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].prefix_events[0].channel, "prediction")
        self.assertEqual(cases[0].positive_event.channel, "prediction_error")

    def test_extracts_image_to_label_cases(self) -> None:
        events = [
            Signal(channel="image", payload="image tokens 1"),
            Signal(channel="label", payload="9:Ankle boot"),
            Signal(channel="image", payload="image tokens 2"),
            Signal(channel="label", payload="0:T-shirt/top"),
        ]

        cases = extract_future_prediction_cases(events, target_channel="label")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].condition, "image_to_label")
        self.assertEqual(cases[0].prefix_events[0].channel, "image")
        self.assertEqual(cases[0].positive_event.payload, "9:Ankle boot")
        self.assertEqual(cases[0].negative_events[0].payload, "0:T-shirt/top")

    def test_renders_prefix_and_continuations_with_signal_stream(self) -> None:
        events = [
            Signal(channel="observation", payload="A.."),
            Signal(channel="action", payload="right"),
            Signal(channel="consequence", payload=".A."),
            Signal(channel="consequence", payload="..B"),
        ]
        case = extract_future_prediction_cases(events)[0]

        prefix = render_future_prediction_prefix(case)
        continuations = render_future_prediction_continuations(case)

        self.assertIn('channel="observation"', prefix)
        self.assertIn('channel="action"', prefix)
        self.assertEqual(len(continuations), 2)
        self.assertIn(".A.", continuations[0])
        self.assertIn("..B", continuations[1])

    def test_extracts_cases_with_structured_text_payloads(self) -> None:
        events = [
            Signal(channel="observation", payload='{"grid":"A.."}'),
            Signal(channel="action", payload='{"move":"right"}'),
            Signal(channel="consequence", payload='{"grid":".A."}'),
            Signal(channel="observation", payload='{"grid":"B.."}'),
            Signal(channel="action", payload='{"move":"left"}'),
            Signal(channel="consequence", payload='{"grid":"..B"}'),
        ]

        cases = extract_future_prediction_cases(events)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].positive_event.payload, '{"grid":".A."}')
        self.assertEqual(cases[0].negative_events[0].payload, '{"grid":"..B"}')

    def test_signal_rendering_renders_structured_text_payloads(self) -> None:
        events = [
            Signal(channel="observation", payload='{"grid":"A.."}'),
            Signal(channel="action", payload='{"move":"right"}'),
            Signal(channel="consequence", payload='{"grid":".A."}'),
            Signal(channel="consequence", payload='{"grid":"..B"}'),
        ]
        case = extract_future_prediction_cases(events)[0]

        prefix = render_future_prediction_prefix(case)

        self.assertIn('{"grid":"A.."}', prefix)
        self.assertIn('{"move":"right"}', prefix)

    def test_payload_rendering_renders_structured_text_payloads(self) -> None:
        events = [
            Signal(channel="observation", payload='{"grid":"A.."}'),
            Signal(channel="action", payload='{"move":"right"}'),
            Signal(channel="consequence", payload='{"grid":".A."}'),
            Signal(channel="consequence", payload='{"grid":"..B"}'),
        ]
        case = extract_future_prediction_cases(events)[0]

        prefix, positive, negatives = render_future_prediction_texts(case, rendering="payload")

        self.assertEqual(prefix, '{"grid":"A.."}\n{"move":"right"}\n')
        self.assertEqual(positive, '{"grid":".A."}\n')
        self.assertEqual(negatives[0], '{"grid":"..B"}\n')

if __name__ == "__main__":
    unittest.main()
