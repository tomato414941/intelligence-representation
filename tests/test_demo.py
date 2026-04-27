import io
import unittest
from contextlib import redirect_stdout

from intrep.demo import main


class DemoTest(unittest.TestCase):
    def test_demo_prints_mixed_gpt_results(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("intrep mixed-gpt demo", text)
        self.assertIn("corpus: documents=", text)
        self.assertIn("environment_symbolic=", text)
        self.assertIn("environment_natural=", text)
        self.assertIn("paired_episodes=", text)
        self.assertIn("training: tokens=", text)
        self.assertIn("initial_loss=", text)
        self.assertIn("final_loss=", text)


if __name__ == "__main__":
    unittest.main()
