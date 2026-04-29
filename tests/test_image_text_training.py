import tempfile
import unittest
from pathlib import Path

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer
from intrep.image_text_evaluation import evaluate_image_text_choices
from intrep.image_text_training import (
    ImageTextExample,
    ImageTextTrainingConfig,
    image_text_examples_from_choices,
    train_image_text_examples,
)


class ImageTextTrainingTest(unittest.TestCase):
    def test_train_image_text_examples_overfits_tiny_example(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            result = train_image_text_examples(
                examples=(
                    ImageTextExample(
                        image_path=image_path,
                        answer_text="b",
                    ),
                ),
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
                config=ImageTextTrainingConfig(max_steps=20, learning_rate=0.05, seed=11),
            )

        self.assertEqual(result.steps, 20)
        self.assertEqual(result.case_count, 1)
        self.assertEqual(len(result.loss_history), 20)
        self.assertGreater(result.initial_loss, result.final_loss)

    def test_training_reduces_correct_choice_evaluation_loss(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            choice_examples = (
                ImageChoiceExample(
                    image_path=image_path,
                    choices=("a", "b"),
                    answer_index=1,
                ),
            )
            before = evaluate_image_text_choices(
                examples=choice_examples,
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
            )
            train_image_text_examples(
                examples=image_text_examples_from_choices(choice_examples),
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
                config=ImageTextTrainingConfig(max_steps=20, learning_rate=0.05, seed=11),
            )
            after = evaluate_image_text_choices(
                examples=choice_examples,
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
            )

        self.assertGreater(before.losses[0][1], after.losses[0][1])

    def test_converts_image_choice_examples_to_image_text_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")

            examples = image_text_examples_from_choices(
                (
                    ImageChoiceExample(
                        image_path=image_path,
                        choices=("a", "b"),
                        answer_index=1,
                    ),
                )
            )

        self.assertEqual(examples, [ImageTextExample(image_path=image_path, answer_text="b")])

    def test_train_image_text_examples_validates_inputs(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )

        with self.assertRaisesRegex(ValueError, "examples"):
            train_image_text_examples(
                examples=(),
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "max_steps"):
                train_image_text_examples(
                    examples=(
                        ImageTextExample(
                            image_path=image_path,
                            answer_text="b",
                        ),
                    ),
                    image_input_layer=image_input,
                    text_model=text_model,
                    tokenizer=tokenizer,
                    prompt="?",
                    config=ImageTextTrainingConfig(max_steps=-1),
                )


if __name__ == "__main__":
    unittest.main()
