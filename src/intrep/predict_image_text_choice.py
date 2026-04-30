from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from intrep.image_conditioned_text_scoring import score_image_conditioned_text_candidates
from intrep.image_io import read_portable_image
from intrep.image_to_text_training import load_image_to_text_checkpoint


@dataclass(frozen=True)
class ImageTextChoicePrediction:
    predicted_index: int
    predicted_choice: str
    choices: tuple[str, ...]
    losses: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted_index": self.predicted_index,
            "predicted_choice": self.predicted_choice,
            "choices": list(self.choices),
            "losses": list(self.losses),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict an image label by scoring text choices with a checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--choice", dest="choices", action="append", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-path", type=Path)
    return parser


def predict_image_text_choice(
    *,
    checkpoint_path: str | Path,
    image_path: str | Path,
    choices: tuple[str, ...],
    device: str = "auto",
) -> ImageTextChoicePrediction:
    checkpoint = load_image_to_text_checkpoint(checkpoint_path, device=device)
    image = torch.tensor(read_portable_image(image_path), dtype=torch.float32)
    losses = score_image_conditioned_text_candidates(
        image_input_layer=checkpoint.image_input_layer,
        text_model=checkpoint.text_model,
        tokenizer=checkpoint.tokenizer,
        image=image,
        prompt="",
        candidates=choices,
    )
    predicted_index = min(range(len(losses)), key=losses.__getitem__)
    return ImageTextChoicePrediction(
        predicted_index=predicted_index,
        predicted_choice=choices[predicted_index],
        choices=choices,
        losses=tuple(losses),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prediction = predict_image_text_choice(
        checkpoint_path=args.checkpoint_path,
        image_path=args.image_path,
        choices=tuple(args.choices),
        device=args.device,
    )
    if args.output_path is not None:
        args.output_path.write_text(json.dumps(prediction.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("intrep image text choice prediction")
    print(f"predicted_index={prediction.predicted_index} predicted_choice={prediction.predicted_choice}")
    print("losses=" + ",".join(f"{loss:.4f}" for loss in prediction.losses))


if __name__ == "__main__":
    main()
