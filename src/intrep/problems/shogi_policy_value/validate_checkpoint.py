from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a shogi policy/value checkpoint entry.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    identity = load_shogi_policy_value_checkpoint_identity(args.checkpoint, device=args.device)
    print(
        json.dumps(
            {
                "checkpoint_path": str(args.checkpoint),
                "checkpoint_id": identity.checkpoint_id,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "schema_version": identity.schema_version,
                "assembly": identity.assembly,
                "assembly_spec_id": identity.assembly_spec_id,
                "input_feature_manifest_hash": identity.input_feature_manifest_hash,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
