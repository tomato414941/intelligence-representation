import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_classification import FASHION_MNIST_LABELS
from intrep.image_text_choice_examples import (
    ImageTextChoiceExample,
    image_text_choice_example_to_record,
    load_image_text_choice_examples_jsonl,
)


class ImageTextChoiceExamplesTest(unittest.TestCase):
    def test_loads_image_text_choice_examples_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            examples = [
                ImageTextChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=9),
                ImageTextChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=0),
            ]
            path = Path(directory) / "fashion.jsonl"
            path.write_text(
                "\n".join(json.dumps(image_text_choice_example_to_record(example)) for example in examples) + "\n",
                encoding="utf-8",
            )

            loaded = load_image_text_choice_examples_jsonl(path)

        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], ImageTextChoiceExample)
        self.assertEqual(loaded[0].answer_text, "Ankle boot")


if __name__ == "__main__":
    unittest.main()
