import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextConfig, CausalTextModel
from intrep.generate_text import generate_text_from_checkpoint, main
from intrep.text_tokenizer import train_byte_pair_tokenizer, text_tokenizer_to_payload


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

    def test_generates_text_from_byte_pair_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            tokenizer = train_byte_pair_tokenizer("hello hello", vocab_size=260)
            _write_checkpoint(
                checkpoint_path,
                vocab_size=tokenizer.vocab_size,
                tokenizer_payload=text_tokenizer_to_payload(tokenizer),
            )

            generated = generate_text_from_checkpoint(
                checkpoint_path=checkpoint_path,
                prompt="hello",
                max_new_tokens=2,
                temperature=0.0,
                device="cpu",
            )

        self.assertTrue(generated.startswith("hello"))


def _write_checkpoint(
    path: Path,
    *,
    vocab_size: int = ByteTokenizer.vocab_size,
    tokenizer_payload: dict[str, object] | None = None,
) -> None:
    config = CausalTextConfig(
        vocab_size=vocab_size,
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
    payload = {
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
    }
    if tokenizer_payload is not None:
        payload["tokenizer"] = tokenizer_payload
    torch.save(payload, path)


if __name__ == "__main__":
    unittest.main()
