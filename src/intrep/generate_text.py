from __future__ import annotations

import argparse
from pathlib import Path

import torch

from intrep.causal_text_model import CausalTextConfig, CausalTextModel
from intrep.text_tokenizer import TextTokenizer, text_tokenizer_from_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a causal text checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    generated = generate_text_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )
    print(generated, end="")


def generate_text_from_checkpoint(
    *,
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    seed: int = 7,
    device: str = "cpu",
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    torch.manual_seed(seed)
    model, tokenizer = load_causal_text_checkpoint(checkpoint_path, device=device)
    token_ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        next_token_id = _next_token_id(
            model=model,
            token_ids=token_ids,
            temperature=temperature,
            device=device,
        )
        token_ids.append(next_token_id)
    return tokenizer.decode(token_ids)


def load_causal_text_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> tuple[CausalTextModel, TextTokenizer]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("schema_version") != "intrep.causal_text_checkpoint.v1":
        raise ValueError("unsupported checkpoint schema")
    model = CausalTextModel(CausalTextConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tokenizer = text_tokenizer_from_payload(payload.get("tokenizer"))
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("checkpoint tokenizer vocab size does not match model vocab size")
    return model, tokenizer


def _next_token_id(
    *,
    model: CausalTextModel,
    token_ids: list[int],
    temperature: float,
    device: str,
) -> int:
    context = token_ids[-model.config.context_length :]
    inputs = torch.tensor([context], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


if __name__ == "__main__":
    main()
