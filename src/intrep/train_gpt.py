from __future__ import annotations

import argparse

from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only GPT on mixed-world data.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = train_mixed_gpt(
        training_config=GPTTrainingConfig(
            max_steps=args.max_steps,
            context_length=args.context_length,
            batch_size=args.batch_size,
        )
    )
    print("intrep mixed-gpt training")
    print(
        f"tokens={result.token_count}"
        f" steps={result.steps}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
    )


if __name__ == "__main__":
    main()
