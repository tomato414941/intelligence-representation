import tempfile
import unittest
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer
from intrep.image_text_evaluation import evaluate_image_text_choices


class ImageTextEvaluationTest(unittest.TestCase):
    def test_evaluates_image_text_choice_examples(self) -> None:
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
        preferred_id = tokenizer.encode("b")[0]
        with torch.no_grad():
            text_model.token_output.output.weight.zero_()
            text_model.token_output.output.bias.zero_()
            text_model.token_output.output.bias[preferred_id] = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            metrics = evaluate_image_text_choices(
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
            )

        self.assertEqual(metrics.case_count, 1)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.predicted_indices, (1,))
        self.assertEqual(len(metrics.losses), 1)
        self.assertEqual(len(metrics.losses[0]), 2)
        self.assertEqual(metrics.to_dict()["predicted_indices"], [1])


if __name__ == "__main__":
    unittest.main()
