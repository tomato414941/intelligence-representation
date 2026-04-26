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
        self.assertIn("frequency_accuracy=0.67", text)
        self.assertIn("state_aware_accuracy=1.00", text)
        self.assertIn("name=held_out_object", text)
        self.assertIn("after_correct=True", text)


if __name__ == "__main__":
    unittest.main()
