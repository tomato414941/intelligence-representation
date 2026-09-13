"""Prepare nested, training-only image supports for the rule-transfer followup."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from intrep.datasets.vision.idx import read_idx_images, read_idx_labels
from intrep.problems.shared_prediction.rule_transfer_data import file_digest, image_training_examples, load_panel
from intrep.problems.shared_prediction.sources import source_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use an empty output directory")
    root = args.data_root.resolve()
    panel, _, _ = load_panel(args.panel, root)
    importlib.import_module("intrep.problems.shared_prediction.record_sources")
    config = next(row for row in source_configs(json.loads(args.recipe.read_text())) if row["name"] == "mnist")
    if (root / config["images"]).resolve() != (root / panel["files"]["training_images"]["path"]).resolve():
        parser.error("the support must use the original MNIST training population")
    files = {name: {"path": config[name], "sha256": file_digest(root / config[name])} for name in ("images", "labels")}
    images, labels = read_idx_images(root / config["images"]), read_idx_labels(root / config["labels"])
    examples = image_training_examples(images, labels, panel["orders"], seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for count in (32, 128, 512):
        selected = examples[:count]
        if selected != image_training_examples(images, labels, panel["orders"], count=count, seed=args.seed):
            raise ValueError("image budgets must be exact nested prefixes")
        before = {(row["digits"][0], row["digits"][1]) for row in selected if row["answer"] == "yes"}
        for middle in range(10):
            before.update((a, b) for a in range(10) for b in range(10)
                          if (a, middle) in before and (middle, b) in before)
        if len(before) != 45:
            raise ValueError("even the smallest support must identify the complete order")
        manifest = {"schema_version": "intrep.rule_transfer_images.v1", "source_split": "train",
                    "seed": args.seed, "order": panel["orders"]["a"], "files": files,
                    "panel_sha256": file_digest(args.panel), "examples": selected}
        path = args.output / f"image-{count:04d}.json"
        with path.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
        summary[count] = {"sha256": file_digest(path), "questions": count, "physical_pairs": count // 2,
                          "unique_training_images": len({index for row in selected for index in row["indices"]}),
                          "class_pairs_observed": len({tuple(sorted(row["digits"])) for row in selected}),
                          "class_relations_identified_by_transitivity": len(before)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
