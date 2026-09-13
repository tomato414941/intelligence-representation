"""Prepare disjoint development/holdout pairs without copying MNIST image payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shared_prediction.rule_transfer_data import historical_files, prepare_panel, text_training_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--mnist", type=Path, default=Path("data/mnist/raw"))
    parser.add_argument("--history", type=Path, action="append", required=True,
                        help="historical report file or directory; repeat to include every known evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--recipe", type=Path, default=Path("configs/question-learning.json"),
                        help="preserve all training populations while reserving the final image panel")
    parser.add_argument("--pairs-per-class-pair", type=int, default=10)
    args = parser.parse_args()
    generated = [args.output, *(args.output.parent / f"text-{name}.json" for name in ("a", "b", "control")),
                 args.output.parent / "development-recipe.json"]
    if any(path.exists() for path in generated):
        parser.error("the panel already exists; do not silently replace a fixed evaluation panel")
    root = args.data_root.resolve()
    mnist = root / args.mnist
    panel = prepare_panel(root, mnist / "t10k-images-idx3-ubyte.gz", mnist / "t10k-labels-idx1-ubyte.gz",
                          mnist / "train-images-idx3-ubyte.gz", historical_files(args.history),
                          seed=args.seed, per_class_pair=args.pairs_per_class_pair)
    recipe = json.loads(args.recipe.read_text())
    mnist_sources = [row for row in recipe["sources"] if row["name"] == "mnist"]
    if len(mnist_sources) != 1:
        parser.error("the recipe must identify exactly one MNIST source")
    evaluation = mnist_sources[0]["evaluation"]
    for name in ("images", "labels"):
        if (root / evaluation[name]).resolve() != (root / panel["files"][name]["path"]).resolve():
            parser.error("the recipe's MNIST evaluation files must match the panel's original files")
    evaluation["evaluation_excluded_indices"] = panel["holdout_excluded_indices"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(panel, indent=2) + "\n")
    for condition in ("a", "b", "control"):
        with (args.output.parent / f"text-{condition}.json").open("x") as handle:
            handle.write(json.dumps({"schema_version": "intrep.rule_transfer_text.v1", "condition": condition,
                                     "examples": text_training_examples(panel["orders"], condition, args.seed)}, indent=2) + "\n")
    with (args.output.parent / "development-recipe.json").open("x") as handle:
        handle.write(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps({"panel": str(args.output), "prior_evaluated_images": len(panel["excluded_indices"]),
                      "historical_reports": len(panel["history"]),
                      "pairs_per_split": {name: len(rows) for name, rows in panel["panels"].items()},
                      "changed_class_pairs": 23, "unchanged_class_pairs": 22}))


if __name__ == "__main__":
    main()
