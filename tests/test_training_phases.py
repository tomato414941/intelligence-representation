import tempfile
import unittest
from pathlib import Path

from intrep.fashion_mnist_vit import ImageChoiceExample
from intrep.image_text_training import ImageTextTrainingConfig
from intrep.language_modeling_training import LanguageModelingTrainingConfig
from intrep.text_examples import LanguageModelingExample
from intrep.training_phases import run_image_text_training_phases


class TrainingPhasesTest(unittest.TestCase):
    def test_runs_image_text_training_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")

            result = run_image_text_training_phases(
                image_examples=(
                    ImageChoiceExample(
                        image_path=image_path,
                        choices=("a", "b"),
                        answer_index=1,
                    ),
                ),
                text_examples=(
                    LanguageModelingExample("label: a"),
                    LanguageModelingExample("label: b"),
                    LanguageModelingExample("label: a"),
                    LanguageModelingExample("label: b"),
                    LanguageModelingExample("label: a"),
                    LanguageModelingExample("label: b"),
                ),
                language_modeling_config=LanguageModelingTrainingConfig(
                    context_length=4,
                    batch_size=2,
                    max_steps=2,
                    learning_rate=0.01,
                    seed=11,
                ),
                image_text_config=ImageTextTrainingConfig(
                    max_steps=10,
                    learning_rate=0.05,
                    seed=13,
                ),
                prompt="?",
            )

        self.assertEqual(result.image_classification.train_case_count, 1)
        self.assertGreater(result.language_modeling.result.token_count, 0)
        self.assertEqual(result.image_text.case_count, 1)
        self.assertGreater(result.image_text.initial_loss, result.image_text.final_loss)


if __name__ == "__main__":
    unittest.main()
