import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep import train_ptm


class TrainPTMCLITest(unittest.TestCase):
    def test_forwards_typed_event_defaults_to_train_gpt(self) -> None:
        captured_argv: list[str] | None = None

        def fake_train_gpt_main(argv):
            nonlocal captured_argv
            captured_argv = list(argv)

        with TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            eval_path = Path(directory) / "eval.jsonl"
            train_path.write_text("", encoding="utf-8")
            eval_path.write_text("", encoding="utf-8")
            with patch.object(train_ptm.train_gpt, "main", fake_train_gpt_main):
                with redirect_stdout(io.StringIO()):
                    train_ptm.main(
                        [
                            "--train-path",
                            str(train_path),
                            "--eval-path",
                            str(eval_path),
                            "--max-steps",
                            "3",
                            "--model-preset",
                            "tiny",
                        ]
                    )

        assert captured_argv is not None
        self.assertIn("--corpus-format", captured_argv)
        self.assertIn("typed-event", captured_argv)
        self.assertIn("--render-format", captured_argv)
        self.assertIn("typed-tags", captured_argv)
        self.assertIn(str(eval_path), captured_argv)


if __name__ == "__main__":
    unittest.main()
