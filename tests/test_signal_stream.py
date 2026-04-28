import unittest

from intrep.signals import Signal
from intrep.signal_stream import render_signal, render_signal_stream


class SignalStreamTest(unittest.TestCase):
    def test_render_signal_uses_signal_tag(self) -> None:
        event = Signal(channel="observation", payload="A..\n.#.")

        rendered = render_signal(event)

        self.assertIn('<SIGNAL channel="observation">', rendered)
        self.assertIn("A..\n.#.\n</SIGNAL>", rendered)

    def test_render_signal_stream_keeps_input_order(self) -> None:
        events = [
            Signal(channel="action", payload="right"),
            Signal(channel="observation", payload="A.."),
        ]

        rendered = render_signal_stream(events)

        self.assertLess(rendered.index("right"), rendered.index("A.."))

    def test_render_signal_rejects_boundary_in_payload(self) -> None:
        event = Signal(channel="text", payload="x</SIGNAL>")

        with self.assertRaisesRegex(ValueError, "must not contain </SIGNAL>"):
            render_signal(event)


if __name__ == "__main__":
    unittest.main()
