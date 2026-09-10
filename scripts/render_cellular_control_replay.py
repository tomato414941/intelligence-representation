#!/usr/bin/env python3
"""Build an offline, self-contained replay from actual control evaluations."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    models = []
    hashes = set()
    for index, path in enumerate(args.evaluations):
        result = json.loads(path.read_text())
        if result["schema_version"] != "intrep.cellular_rule_control.v1":
            raise ValueError("not a cellular control evaluation")
        config = result["training_config"]["model"]
        if (config["height"], config["width"]) != (6, 6):
            raise ValueError("this replay presents the six-by-six control experiment")
        hashes.add(result["task_sha256"])
        final = result["summaries"][-1]["methods"]
        worlds = [{"steps": [row["trace"] | {"context_count": row["context_count"]}
                              for row in world["rounds"]]}
                  for world in result["per_rule"] if "trace" in world["rounds"][0]]
        models.append({"name": f"モデル {chr(65 + index)}", "seed": result["training_config"]["model_seed"],
                       "world_count": len(result["per_rule"]),
                       "trials": result["trials_per_rule"], "worlds": worlds,
                       "experience_rate": final["experience"]["optimal_action_rate"]["mean"],
                       "forgetful_rate": final["forgetful"]["optimal_action_rate"]["mean"]})
    if len(hashes) != 1:
        raise ValueError("compared models must have identical control tasks")
    data = {"models": models}
    if args.figure:
        data["figure"] = "data:image/png;base64," + base64.b64encode(args.figure.read_bytes()).decode()
    template = Path(__file__).with_name("cellular_control_replay.html").read_text()
    output = template.replace("__CONTROL_DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
