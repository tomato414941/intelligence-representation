from __future__ import annotations

import copy
from pathlib import Path

from intrep.problems.shared_prediction.sources import source_configs


def evaluation_recipe(recipe: dict) -> dict:
    result = copy.deepcopy(recipe)
    result["sources"] = []
    for source in recipe["sources"]:
        if not isinstance(source.get("evaluation"), dict) or not source["evaluation"]:
            raise ValueError("every source must declare an evaluation split")
        result["sources"].append({**source, **source["evaluation"]})
    return result


def validate_recipe(recipe: dict, root: Path) -> None:
    train = source_configs(recipe)
    evaluation = source_configs(evaluation_recipe(recipe))
    if {row["name"] for row in train} != {row["name"] for row in evaluation}:
        raise ValueError("evaluation must preserve every training source identity")
    patches = {row["patch_size"] for row in train if row["kind"] in {"idx", "cifar10", "native"}}
    if len(patches) > 1:
        raise ValueError("sources sharing the rgb head must agree on its patch size")
    intervals = {"train": [], "evaluation": []}
    for split, rows in (("train", train), ("evaluation", evaluation)):
        for row in rows:
            if any(key in row for key in ("limit", "max_examples", "max_games")):
                raise ValueError("joint recipes traverse complete declared populations without subset limits")
            if row["kind"] == "native":
                expected = "train" if split == "train" else "validation"
                if row.get("split", "train") != expected:
                    raise ValueError("native sources must use separate train and validation worlds")
                continue
            names = ([row["path"]] if "path" in row else
                     [row["images"], row["labels"]] if "images" in row else row.get("batches", []))
            for name in names:
                path = (root / name).resolve()
                start, end = row.get("byte_start", 0), row.get("byte_end", path.stat().st_size)
                if not 0 <= start < end <= path.stat().st_size:
                    raise ValueError("source file range is empty or outside its file")
                intervals[split].append((path, start, end))
    for train_path, train_start, train_end in intervals["train"]:
        for eval_path, eval_start, eval_end in intervals["evaluation"]:
            if train_path == eval_path and max(train_start, eval_start) < min(train_end, eval_end):
                raise ValueError("training and evaluation source ranges overlap")


def validate_extension(previous: dict, following: dict) -> None:
    if previous.get("defaults", {}) != following.get("defaults", {}):
        raise ValueError("extension must preserve the previous shared-head and sampling defaults")
    old = {row["name"]: row for row in previous["sources"]}
    new = {row["name"]: row for row in following["sources"]}
    if any(new.get(name) != row for name, row in old.items()):
        raise ValueError("extension must retain every existing source and its configuration")
