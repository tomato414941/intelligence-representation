import unittest

from intrep.typed_events import TypedEvent
from intrep.typed_stream import render_typed_event, render_typed_stream


class TypedStreamTest(unittest.TestCase):
    def test_render_typed_event_uses_stable_event_tag(self) -> None:
        event = TypedEvent(
            id="ep1_obs0",
            role="observation",
            modality="grid",
            content="A..\n.#.",
            episode_id="ep1",
            time_index=0,
        )

        rendered = render_typed_event(event)

        self.assertIn(
            '<EVENT id="ep1_obs0" role="observation" modality="grid" episode="ep1" t="0">',
            rendered,
        )
        self.assertIn("A..\n.#.\n</EVENT>", rendered)

    def test_render_typed_stream_orders_by_episode_time_and_id(self) -> None:
        events = [
            TypedEvent(id="ep1_t2", role="action", modality="grid_action", content="right", episode_id="ep1", time_index=2),
            TypedEvent(id="ep1_t1", role="observation", modality="grid", content="A..", episode_id="ep1", time_index=1),
        ]

        rendered = render_typed_stream(events)

        self.assertLess(rendered.index("ep1_t1"), rendered.index("ep1_t2"))

    def test_render_typed_event_rejects_boundary_in_content(self) -> None:
        event = TypedEvent(id="bad", role="text", modality="text", content="x</EVENT>")

        with self.assertRaisesRegex(ValueError, "must not contain </EVENT>"):
            render_typed_event(event)

    def test_render_typed_event_rejects_whitespace_in_tag_attributes(self) -> None:
        event = TypedEvent(id="bad id", role="text", modality="text", content="x")

        with self.assertRaisesRegex(ValueError, "attributes must not contain whitespace"):
            render_typed_event(event)


if __name__ == "__main__":
    unittest.main()
