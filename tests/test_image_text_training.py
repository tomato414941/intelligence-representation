import tempfile
import unittest
from pathlib import Path

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer
from intrep.image_text_training import (
    ImageTextTrainingConfig,
    train_image_text_choices,
)


class ImageTextTrainingTest(unittest.TestCase):
    def test_train_image_text_choices_overfits_tiny_example(self) -> None:
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
            result = train_image_text_choices(
                examples=(
                    ImageChoiceExample(
                        image_path=image_path,
                        choices=("a", "b"),
                        answer_index=1,
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

    def test_train_image_text_choices_validates_inputs(self) -> None:
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
            train_image_text_choices(
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
                train_image_text_choices(
                    examples=(
                        ImageChoiceExample(
                            image_path=image_path,
                            choices=("a", "b"),
                            answer_index=1,
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
