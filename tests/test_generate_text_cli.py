import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextConfig, CausalTextModel
from intrep.generate_text import generate_text_from_checkpoint, main


class GenerateTextCLITest(unittest.TestCase):
    def test_generates_text_from_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            _write_checkpoint(checkpoint_path)

            generated = generate_text_from_checkpoint(
                checkpoint_path=checkpoint_path,
                prompt="A",
                max_new_tokens=3,
                temperature=0.0,
                device="cpu",
            )

        self.assertEqual(len(generated), 4)
        self.assertTrue(generated.startswith("A"))

    def test_main_prints_generated_text(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            _write_checkpoint(checkpoint_path)

            with redirect_stdout(output):
                main(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--prompt",
                        "A",
                        "--max-new-tokens",
                        "2",
                        "--temperature",
                        "0",
                    ]
                )

        self.assertEqual(len(output.getvalue()), 3)
        self.assertTrue(output.getvalue().startswith("A"))


def _write_checkpoint(path: Path) -> None:
    config = CausalTextConfig(
        vocab_size=ByteTokenizer.vocab_size,
        context_length=8,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
    )
    model = CausalTextModel(config)
    with torch.no_grad():
        model.token_output.output.weight.zero_()
        model.token_output.output.bias.zero_()
        model.token_output.output.bias[ord("B")] = 1.0
    torch.save(
        {
            "schema_version": "intrep.causal_text_checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": config.vocab_size,
                "context_length": config.context_length,
                "embedding_dim": config.embedding_dim,
                "num_heads": config.num_heads,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
            },
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
