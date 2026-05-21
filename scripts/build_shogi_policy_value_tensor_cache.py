from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.output_space import (
    shogi_policy_value_output_space_for_assembly_spec,
)
from intrep.problems.shogi_policy_value.tensor_cache import build_shogi_policy_value_tensor_cache
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS,
    shogi_policy_value_input_for_assembly_spec_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tensor cache for a shogi policy/value data selection.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--assembly-spec",
        choices=SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS,
        required=True,
    )
    parser.add_argument("--shard-examples", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_module = shogi_policy_value_input_for_assembly_spec_id(args.assembly_spec)
    output_space = shogi_policy_value_output_space_for_assembly_spec(args.assembly_spec)
    summary = build_shogi_policy_value_tensor_cache(
        data_selection_path=args.data_selection,
        output_path=args.out,
        output_space=output_space,
        input_module=input_module,
        shard_examples=args.shard_examples,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
