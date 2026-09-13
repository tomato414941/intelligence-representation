"""Score paired A/B rule interventions without treating four answers as four samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shared_prediction.rule_transfer import compare_counterfactuals
from intrep.problems.shared_prediction.rule_transfer_data import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file")
    a, b = json.loads(args.a.read_text()), json.loads(args.b.read_text())
    result = {"schema_version": "intrep.rule_transfer_comparison.v1", "panel_sha256": a["panel_sha256"],
              "split": a["split"], "reports_sha256": {"a": file_digest(args.a), "b": file_digest(args.b)},
              "prerequisite_gates_passed": a["transfer_interpretable"] and b["transfer_interpretable"],
              "counterfactuals": compare_counterfactuals(a, b),
              "scope": "Descriptive paired scores. A causal transfer claim additionally requires verified common initial weights, matched background source traces, and no image supervision for the new rule.",
              "replication": "Image pairs are nested within 45 digit-class pairs. The four answers share images and rules; they are not independent trials. One order pair is not replication across independently taught rules."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
