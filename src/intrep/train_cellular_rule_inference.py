"""Train the bounded unseen-cellular-rule experiment."""

import argparse
from pathlib import Path

from intrep.problems.cellular_rule_inference.training import (
    RuleInferenceTrainingConfig,
    train,
)
from intrep.representation.assemblies.cellular_rule_inference import (
    CellularRuleInferenceModelConfig,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-rule-count", type=int, default=256)
    parser.add_argument("--rule-seed", type=int, default=1701)
    parser.add_argument("--data-seed", type=int, default=3101)
    parser.add_argument("--model-seed", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--observation-noise", action="store_true", help="Mix 0/5/10/20 percent output-cell noise during training.")
    parser.add_argument("--rule-change-probability", type=float, default=0.0,
                        help="Chance of an old-rule prefix when at least two demonstrations are present.")
    args = parser.parse_args(argv)
    model = CellularRuleInferenceModelConfig(height=args.height, width=args.width, embedding_dim=args.embedding_dim,
                                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers)
    config = RuleInferenceTrainingConfig(
        model=model, train_rule_count=args.train_rule_count, rule_seed=args.rule_seed, data_seed=args.data_seed,
        model_seed=args.model_seed, max_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        observation_noise=args.observation_noise, rule_change_probability=args.rule_change_probability,
    )
    print(train(config, args.run_dir, device=args.device, resume=args.resume))


if __name__ == "__main__":
    main()
