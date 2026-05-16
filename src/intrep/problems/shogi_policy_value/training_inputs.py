from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    load_shogi_policy_value_data_selection,
)


@dataclass(frozen=True)
class ShogiPolicyValueTrainingInputs:
    data_selection_path: Path
    data_selection: ShogiPolicyValueDataSelection
    tensor_cache_path: Path | None = None

    def artifact_paths(self) -> tuple[Path, ...]:
        paths = {self.data_selection_path}
        for source in (
            *self.data_selection.train_sources,
            *self.data_selection.eval_sources,
            *self.data_selection.analysis_sources,
        ):
            paths.add(source.path)
        if self.tensor_cache_path is not None:
            paths.add(self.tensor_cache_path)
        return tuple(sorted(paths))


def load_shogi_policy_value_training_inputs(
    *,
    data_selection_path: Path,
    tensor_cache_path: Path | None = None,
) -> ShogiPolicyValueTrainingInputs:
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    if not data_selection_path.exists():
        raise FileNotFoundError(f"data selection not found: {data_selection_path}")
    if tensor_cache_path is not None and not tensor_cache_path.is_dir():
        raise FileNotFoundError(f"tensor cache not found: {tensor_cache_path}")
    inputs = ShogiPolicyValueTrainingInputs(
        data_selection_path=data_selection_path,
        data_selection=data_selection,
        tensor_cache_path=tensor_cache_path,
    )
    for path in inputs.artifact_paths():
        if not path.exists():
            raise FileNotFoundError(f"training input artifact not found: {path}")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="List shogi policy/value training input artifact paths.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path)
    args = parser.parse_args()

    inputs = load_shogi_policy_value_training_inputs(
        data_selection_path=args.data_selection,
        tensor_cache_path=args.tensor_cache,
    )
    for path in inputs.artifact_paths():
        print(path)


if __name__ == "__main__":
    main()
