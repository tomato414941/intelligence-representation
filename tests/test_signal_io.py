import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.signal_io import (
    first_json_record,
    load_signals_jsonl_v2,
    write_signals_jsonl_v2,
)
from intrep.signals import PayloadRef, Signal


class SignalIoTest(unittest.TestCase):
    def test_jsonl_v2_round_trips_text_signals_without_rendering(self) -> None:
        events = [Signal(channel="observation", payload="<obs> raw")]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"

            write_signals_jsonl_v2(path, events)

            loaded = load_signals_jsonl_v2(path)

        self.assertEqual(loaded, events)

    def test_jsonl_v2_round_trips_payload_refs_as_data(self) -> None:
        events = [
            Signal(
                channel="image",
                payload=PayloadRef(
                    uri="dataset://images/frame-1.png",
                    media_type="image/png",
                    sha256="b" * 64,
                    size_bytes=456,
                ),
            )
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"

            write_signals_jsonl_v2(path, events)

            loaded = load_signals_jsonl_v2(path)

        self.assertEqual(loaded, events)

    def test_jsonl_loader_accepts_legacy_role_content_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            path.write_text(
                '{"role":"action","content":"move east"}\n',
                encoding="utf-8",
            )

            loaded = load_signals_jsonl_v2(path)

        self.assertEqual(loaded, [Signal(channel="action", payload="move east")])

    def test_first_json_record_skips_blank_lines(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"
            path.write_text(
                '\n\n{"channel":"observation","payload":"A.."}\n',
                encoding="utf-8",
            )

            record = first_json_record(path)

        self.assertEqual(record, {"channel": "observation", "payload": "A.."})

    def test_jsonl_loader_rejects_invalid_payload_ref_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"
            path.write_text(
                '{"channel":"image","payload_ref":{"uri":"dataset://images/frame-1.png"}}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing payload_ref fields: media_type"):
                load_signals_jsonl_v2(path)


if __name__ == "__main__":
    unittest.main()
