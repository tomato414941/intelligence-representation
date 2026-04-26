import io
import unittest
from contextlib import redirect_stdout

from intrep.demo import main


class DemoTest(unittest.TestCase):
    def test_demo_prints_core_prototype_results(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("intrep prototype demo", text)
        self.assertIn("frequency_accuracy=0.53", text)
        self.assertIn("state_aware_accuracy=1.00", text)
        self.assertIn("transformer_ready_accuracy=", text)
        self.assertIn("sequence_feature_accuracy=0.40", text)
        self.assertIn("name=missing_link", text)
        self.assertIn("state_aware_unsupported=1.00", text)
        self.assertIn("after_correct=True", text)


if __name__ == "__main__":
    unittest.main()
