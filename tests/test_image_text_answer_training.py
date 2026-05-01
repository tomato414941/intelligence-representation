import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.image_text_answer_training import (
    ImageTextAnswerExample,
    ImageTextAnswerTrainingConfig,
    generate_image_text_answer,
    image_text_answer_example_to_record,
    load_image_text_answer_examples_jsonl,
    train_image_text_answer_model,
)
from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageTextAnswerTrainingTest(unittest.TestCase):
    def test_trains_image_prompt_to_answer_tokens(self) -> None:
        with TemporaryDirectory() as directory:
            examples = _write_examples(Path(directory))
            result = train_image_text_answer_model(
                train_examples=examples,
                tokenizer_corpus="T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot",
                config=ImageTextAnswerTrainingConfig(
                    text_context_length=32,
                    image_patch_size=1,
                    max_steps=4,
                    batch_size=2,
                    learning_rate=0.01,
                    seed=17,
                    model_preset="tiny",
                    device="cpu",
                    tokenizer_vocab_size=270,
                ),
            )
            generated = generate_image_text_answer(
                model=result.model,
                tokenizer=result.tokenizer,
                image=torch.zeros((2, 2), dtype=torch.float32),
                prompt="answer: ",
                max_new_tokens=2,
            )

        self.assertIsInstance(result.model, SharedMultimodalModel)
        self.assertEqual(result.metrics.train_case_count, 2)
        self.assertGreater(result.metrics.train_initial_loss, 0.0)
        self.assertGreater(result.metrics.train_final_loss, 0.0)
        self.assertIsInstance(generated, str)

    def test_loads_image_text_answer_examples_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            examples = _write_examples(Path(directory))
            path = Path(directory) / "answers.jsonl"
            path.write_text(
                "\n".join(json.dumps(image_text_answer_example_to_record(example)) for example in examples) + "\n",
                encoding="utf-8",
            )

            loaded = load_image_text_answer_examples_jsonl(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].prompt, "answer: ")
        self.assertEqual(loaded[0].answer_text, "Ankle boot")

    def test_load_image_text_answer_examples_jsonl_rejects_empty_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain at least one example"):
                load_image_text_answer_examples_jsonl(path)

    def test_load_image_text_answer_examples_jsonl_rejects_unsupported_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "image_path": "a.pgm",
                        "prompt": "answer: ",
                        "answer_text": "Ankle boot",
                        "choices": ["Ankle boot"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported fields: choices"):
                load_image_text_answer_examples_jsonl(path)


def _write_examples(root: Path) -> list[ImageTextAnswerExample]:
    image_a = root / "a.pgm"
    image_b = root / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    return [
        ImageTextAnswerExample(image_path=image_a, prompt="answer: ", answer_text="Ankle boot"),
        ImageTextAnswerExample(image_path=image_b, prompt="answer: ", answer_text="T-shirt/top"),
    ]


if __name__ == "__main__":
    unittest.main()
