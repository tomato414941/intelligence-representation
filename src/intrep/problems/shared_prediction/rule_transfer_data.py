"""Fixed MNIST panels for the temporary text-to-image order-transfer experiment."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path

import numpy as np

from intrep.datasets.vision.idx import read_idx_images, read_idx_labels

PANEL_SCHEMA = "intrep.rule_transfer_panel.v1"
DIGITS = tuple(range(10))
CLASS_PAIRS = tuple(itertools.combinations(DIGITS, 2))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def precedes(order, first: int, second: int) -> bool:
    return order.index(first) < order.index(second)


def make_orders(seed: int, changed_pairs: int = 23) -> dict:
    if not 0 < changed_pairs < len(CLASS_PAIRS):
        raise ValueError("the intervention must include both changed and unchanged class pairs")
    generator = random.Random(seed)
    first = generator.sample(DIGITS, len(DIGITS))
    for _ in range(10000):
        second = generator.sample(DIGITS, len(DIGITS))
        if sum(precedes(first, a, b) != precedes(second, a, b) for a, b in CLASS_PAIRS) == changed_pairs:
            return {"old": list(DIGITS), "a": first, "b": second}
    raise ValueError("could not construct the requested counterfactual orders")


def text_training_examples(orders, condition, seed=47):
    if condition not in ("a", "b", "control"):
        raise ValueError("choose counterfactual a/b or old-rule rehearsal")
    rule, order = ("old", orders["old"]) if condition == "control" else ("new", orders[condition])
    pairs = list(itertools.permutations(DIGITS, 2))
    random.Random(seed).shuffle(pairs)
    return [{"id": f"text-{a}-{b}", "rule": rule, "digits": [a, b],
             "answer": "yes" if precedes(order, a, b) else "no"} for a, b in pairs]


def image_training_examples(images, labels, orders, *, count=512, seed=71):
    """Build nested, balanced training supports that identify the complete order."""
    if count not in (32, 128, 512) or len(images) != len(labels) or set(map(int, labels)) != set(DIGITS):
        raise ValueError("image tuition requires 32/128/512 questions and all ten training classes")
    order = orders["a"]
    if sorted(order) != list(DIGITS):
        raise ValueError("image tuition needs the prepared order A")
    generator = random.Random(seed)
    pools = {digit: list(map(int, np.flatnonzero(np.asarray(labels) == digit))) for digit in DIGITS}
    for pool in pools.values():
        generator.shuffle(pool)
    chain = [tuple(sorted(pair)) for pair in zip(order, order[1:])]
    remaining = [pair for pair in CLASS_PAIRS if pair not in chain]
    generator.shuffle(remaining)
    pairs = chain + remaining
    while len(pairs) < count // 2:
        cycle = list(CLASS_PAIRS)
        generator.shuffle(cycle)
        pairs.extend(cycle)
    used_hashes = set()

    def take(digit):
        while pools[digit]:
            index = pools[digit].pop()
            digest = hashlib.sha256(np.asarray(images[index]).tobytes()).hexdigest()
            if digest not in used_hashes:
                used_hashes.add(digest)
                return index, digest
        raise ValueError("not enough distinct training pixels for the image tuition")

    examples = []
    for number, digits in enumerate(pairs[:count // 2]):
        selected = [take(digit) for digit in digits]
        for orientation, direction in ((0, 1), (1, -1)):
            ordered_digits = list(digits[::direction])
            examples.append({"id": f"image-order-{number:04d}-{orientation}",
                             "pair_id": f"image-order-{number:04d}", "orientation": orientation,
                             "indices": [row[0] for row in selected[::direction]], "digits": ordered_digits,
                             "image_sha256": [row[1] for row in selected[::direction]],
                             "answer": "yes" if precedes(order, *ordered_digits) else "no"})
    return examples


def validate_image_manifest(manifest, images, labels, orders):
    if (manifest.get("schema_version") != "intrep.rule_transfer_images.v1"
            or manifest.get("source_split") != "train" or manifest.get("order") != orders["a"]
            or type(manifest.get("seed")) is not int):
        raise ValueError("image tuition must declare the training split and prepared order A")
    expected = image_training_examples(images, labels, orders, count=len(manifest["examples"]), seed=manifest["seed"])
    if manifest["examples"] != expected:
        raise ValueError("image tuition differs from its deterministic training-only support")


def historical_indices(payload) -> set[int]:
    """Read primary and partner indices only inside explicitly named MNIST results."""
    found = set()

    def visit(value, mnist=False):
        if isinstance(value, dict):
            schema = value.get("schema_version")
            if schema in ("intrep.rule_transfer_evaluation.v1", "intrep.rule_transfer_prerequisites.v1",
                          "intrep.rule_transfer_image_prerequisites.v1"):
                indices = [row["index"] for row in value["digit_readouts"]]
                relation_rows = value["rows"] if schema == "intrep.rule_transfer_evaluation.v1" else value["old_image_rows"]
                indices += [index for row in relation_rows for index in row["indices"]]
                if any(type(index) is not int or index < 0 for index in indices):
                    raise ValueError("historical rule-transfer indices must be nonnegative integers")
                found.update(indices)
            mnist = mnist or value.get("source") == "mnist" or value.get("name") == "mnist"
            if mnist:
                for name in ("key", "record_key", "group"):
                    match = re.match(r"^image:(\d+)(?:/|$)", str(value.get(name, "")))
                    if match:
                        found.add(int(match[1]))
                for index in value.get("record_indices", []):
                    if type(index) is not int or index < 0:
                        raise ValueError("historical MNIST indices must be nonnegative integers")
                    found.add(index)
            for name, child in value.items():
                visit(child, mnist or name == "mnist")
        elif isinstance(value, list):
            for child in value:
                visit(child, mnist)

    visit(payload)
    return found


def historical_files(directories) -> list[Path]:
    paths = set()
    for directory in directories:
        directory = Path(directory)
        if directory.is_file():
            paths.add(directory)
        elif directory.is_dir():
            paths.update(path for path in directory.rglob("*.json")
                         if path.name in ("evaluation-panel.json", "background-panel.json", "result.json", "comparison.json")
                         or path.name.endswith(("-development.json", "-holdout.json"))
                         or {"evaluation", "background", "prerequisites"}.intersection(path.relative_to(directory).parts))
        else:
            raise FileNotFoundError(directory)
    return sorted(paths)


def make_pairs(images, labels, orders, *, excluded=(), blocked_hashes=(), per_class_pair=10, seed=47):
    if per_class_pair < 1 or len(images) != len(labels) or set(map(int, labels)) != set(DIGITS):
        raise ValueError("panels require all ten digit classes, matching images and labels, and a positive pair count")
    excluded = set(excluded)
    if any(type(index) is not int or not 0 <= index < len(labels) for index in excluded):
        raise ValueError("a historical index is outside the declared MNIST test population")
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    used_hashes = set(blocked_hashes) | {hashes[index] for index in excluded}
    generator = random.Random(seed)
    pools = {digit: [int(index) for index in np.flatnonzero(labels == digit)
                     if int(index) not in excluded and hashes[int(index)] not in used_hashes] for digit in DIGITS}
    for pool in pools.values():
        generator.shuffle(pool)

    def take(digit):
        while pools[digit]:
            index = pools[digit].pop()
            if hashes[index] not in used_hashes:
                used_hashes.add(hashes[index])
                return index
        raise ValueError(f"not enough distinct unused images of digit {digit}")

    panels = {}
    for split in ("development", "holdout"):
        rows = []
        for a, b in CLASS_PAIRS:
            for repeat in range(per_class_pair):
                indices = [take(a), take(b)]
                rows.append({"id": f"{split}-{a}-{b}-{repeat:03d}", "indices": indices,
                             "digits": [a, b], "image_sha256": [hashes[index] for index in indices],
                             "changed": precedes(orders["a"], a, b) != precedes(orders["b"], a, b)})
        generator.shuffle(rows)
        panels[split] = rows
    return panels


def prepare_panel(root: Path, images_path: Path, labels_path: Path, train_images_path: Path,
                  history, *, seed=47, per_class_pair=10) -> dict:
    root = root.resolve()
    files = {}
    for name, path in (("images", images_path), ("labels", labels_path), ("training_images", train_images_path)):
        path = path.resolve()
        files[name] = {"path": str(path.relative_to(root)), "sha256": file_digest(path), "bytes": path.stat().st_size}
    images, labels = read_idx_images(images_path), read_idx_labels(labels_path)
    excluded, records = set(), []
    for path in sorted(set(history)):
        indices = historical_indices(json.loads(path.read_text()))
        if indices:
            excluded.update(indices)
            records.append({"path": str(path.resolve().relative_to(root)), "sha256": file_digest(path),
                            "indices": sorted(indices)})
    if not records:
        raise ValueError("no historical MNIST evaluation records were supplied")
    training_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in read_idx_images(train_images_path)}
    orders = make_orders(seed)
    result = {"schema_version": PANEL_SCHEMA, "seed": seed, "orders": orders, "files": files,
              "per_class_pair": per_class_pair, "excluded_indices": sorted(excluded), "history": records,
              "history_scope": "Local historical evaluation records explicitly supplied during panel preparation.",
              "panels": make_pairs(images, labels, orders, excluded=excluded, blocked_hashes=training_hashes,
                                   per_class_pair=per_class_pair, seed=seed + 1)}
    holdout_hashes = {digest for row in result["panels"]["holdout"] for digest in row["image_sha256"]}
    result["holdout_excluded_indices"] = [index for index, image in enumerate(images)
                                          if hashlib.sha256(image.tobytes()).hexdigest() in holdout_hashes]
    validate_panel(result, images, labels)
    return result


def validate_panel(panel, images=None, labels=None):
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("unknown order-transfer panel schema")
    if type(panel["per_class_pair"]) is not int or panel["per_class_pair"] < 1:
        raise ValueError("each class pair needs a positive number of image pairs")
    orders = panel["orders"]
    if set(orders) != {"old", "a", "b"} or any(sorted(order) != list(DIGITS) for order in orders.values()):
        raise ValueError("each order must be a permutation of all ten digits")
    changed = sum(precedes(orders["a"], a, b) != precedes(orders["b"], a, b) for a, b in CLASS_PAIRS)
    if changed != 23:
        raise ValueError("the fixed design requires 23 changed and 22 unchanged class pairs")
    used_indices, used_hashes, ids = set(), set(), set()
    excluded = set(panel["excluded_indices"])
    if set(panel["panels"]) != {"development", "holdout"}:
        raise ValueError("both development and holdout panels are required")
    if not {index for row in panel["panels"]["holdout"] for index in row["indices"]}.issubset(
            panel["holdout_excluded_indices"]):
        raise ValueError("ordinary development evaluation must exclude every holdout image")
    for split, rows in panel["panels"].items():
        counts = {pair: 0 for pair in CLASS_PAIRS}
        for row in rows:
            pair = tuple(row["digits"])
            if pair not in counts or row["id"] in ids or len(row["indices"]) != 2 or len(row["image_sha256"]) != 2:
                raise ValueError("invalid or repeated image pair")
            ids.add(row["id"])
            counts[pair] += 1
            if row["changed"] != (precedes(orders["a"], *pair) != precedes(orders["b"], *pair)):
                raise ValueError("counterfactual stratum disagrees with the orders")
            for index, digit, digest in zip(row["indices"], pair, row["image_sha256"]):
                if index in used_indices or index in excluded or digest in used_hashes:
                    raise ValueError("panel images overlap each other or prior evaluation")
                used_indices.add(index)
                used_hashes.add(digest)
                if images is not None:
                    if not 0 <= index < len(images) or hashlib.sha256(images[index].tobytes()).hexdigest() != digest:
                        raise ValueError("panel image bytes changed")
                if labels is not None and int(labels[index]) != digit:
                    raise ValueError("panel label changed")
        if any(count != panel["per_class_pair"] for count in counts.values()):
            raise ValueError(f"{split} is not balanced over all 45 class pairs")


def load_panel(path: Path, root: Path):
    panel = json.loads(path.read_text())
    for name in ("images", "labels", "training_images"):
        row = panel["files"][name]
        if file_digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"the panel's {name} source file changed")
    images = read_idx_images(root / panel["files"]["images"]["path"])
    labels = read_idx_labels(root / panel["files"]["labels"]["path"])
    validate_panel(panel, images, labels)
    return panel, images, labels
