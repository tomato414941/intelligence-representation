import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from intrep import prepare_fineweb_edu_text


class PrepareFineWebEduTextCLITest(unittest.TestCase):
    def test_writes_text_slice_until_byte_limit(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "fineweb.txt"

            result = prepare_fineweb_edu_text.write_text_slice(
                records=(
                    {"text": "alpha"},
                    {"text": ""},
                    {"text": "beta gamma"},
                ),
                output_path=output_path,
                max_bytes=12,
            )

            payload = output_path.read_bytes()

        self.assertEqual(payload, b"alpha\n\nbeta ")
        self.assertEqual(result.document_count, 2)
        self.assertEqual(result.byte_count, 12)

    def test_rejects_non_positive_byte_limit(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "max_bytes must be positive"):
                prepare_fineweb_edu_text.write_text_slice(
                    records=(),
                    output_path=Path(directory) / "fineweb.txt",
                    max_bytes=0,
                )

    def test_truncation_keeps_utf8_valid(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "fineweb.txt"

            result = prepare_fineweb_edu_text.write_text_slice(
                records=({"text": "alpha あ"},),
                output_path=output_path,
                max_bytes=8,
            )

            payload = output_path.read_text(encoding="utf-8")

        self.assertEqual(payload, "alpha ")
        self.assertEqual(result.byte_count, 6)

    def test_cli_uses_streaming_loader(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "fineweb.txt"
            records = ({"text": "first document"}, {"text": "second document"})

            with mock.patch.object(
                prepare_fineweb_edu_text,
                "load_streaming_dataset",
                return_value=records,
            ) as load_streaming_dataset:
                with redirect_stdout(output):
                    prepare_fineweb_edu_text.main(
                        [
                            "--output-path",
                            str(output_path),
                            "--max-bytes",
                            "100",
                            "--dataset-name",
                            "test/dataset",
                            "--split",
                            "train",
                        ]
                    )

            payload = output_path.read_text(encoding="utf-8")

        load_streaming_dataset.assert_called_once_with(
            dataset_name="test/dataset",
            dataset_config=None,
            split="train",
        )
        self.assertIn("first document", payload)
        self.assertIn("second document", payload)
        self.assertIn("intrep prepare fineweb edu text", output.getvalue())
        self.assertIn("documents=2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
