"""Prepare broader bilingual replay without replacing existing source populations.

Run with ``uv run --with pyarrow python scripts/prepare_replay_coverage.py ...``.
Inputs are pinned Aya parquet and the complete LLM-jp v3 Japanese Wikipedia
download, with upstream revisions and checksums in their accompanying manifests.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from intrep.problems.shared_prediction.streams import file_identity


def prompt_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def prompt_group(value):
    return hashlib.sha256(prompt_key(value).encode()).hexdigest()


def aya_rows(rows, forbidden, existing_training=()):
    """Keep both languages; keep alternate answers to one prompt in one split."""
    populations = {(language, split): [] for language in ("en", "ja") for split in ("train", "validation")}
    counts = Counter()
    for index, row in enumerate(rows):
        language = {"eng": "en", "jpn": "ja"}.get(row["language_code"])
        if language is None:
            continue
        counts[f"selected_{language}"] += 1
        prompt, answer = row["inputs"], row["targets"]
        if not prompt or not answer or not prompt.strip() or not answer.strip():
            counts["empty"] += 1
            continue
        if prompt_key(prompt) in forbidden:
            counts["evaluation_prompt_overlap"] += 1
            continue
        group = prompt_group(prompt)
        split = "validation" if int(group[:16], 16) % 20 == 0 else "train"
        if split == "validation" and prompt_key(prompt) in existing_training:
            split = "train"
            counts["validation_prompt_already_in_existing_training"] += 1
        populations[language, split].append({
            "id": f"aya:{language}:{index}", "group_id": group,
            "source": "CohereLabs/aya_dataset", "language": language,
            "annotation_type": row["annotation_type"],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        })
        counts[f"{language}_{split}_{row['annotation_type']}"] += 1
    for rows in populations.values():
        random.Random(16092026).shuffle(rows)
    return populations, dict(counts)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_wikipedia(raw, output, manifest):
    files = [raw / Path(row["path"]).name for row in manifest["files"]]
    validation = [path for path in files if path.name.startswith("validation_")]
    training = sorted((path for path in files if path.name.startswith("train_")),
                      key=lambda path: int(path.name.split("_")[1].split(".")[0]))
    if len(validation) != 1 or len(training) != 14:
        raise ValueError("Expected the complete pinned subset: fourteen train files and one validation file.")
    for path, entry in zip(files, manifest["files"]):
        actual = file_identity(path)
        if (actual["sha256"], actual["bytes"]) != (entry["sha256"], entry["bytes"]):
            raise ValueError(f"Wikipedia input changed: {path.name}")
    counts = Counter()
    heldout_ids, heldout_text = set(), set()
    temporary = output / ".wiki-shuffle"
    temporary.mkdir()
    with gzip.open(output / "wiki-articles.jsonl.gz", "wt") as provenance:
        def emit(handle, row):
            start = handle.tell()
            handle.write(row.pop("text").encode() + b"\n\n")
            provenance.write(json.dumps({**row, "start": start, "end": handle.tell()}, ensure_ascii=False) + "\n")

        with (output / "wiki-validation.txt").open("wb") as handle:
            with gzip.open(validation[0], "rt") as source:
                for index, line in enumerate(source):
                    row = json.loads(line)
                    text = row["text"].strip()
                    if not text:
                        raise ValueError("The published validation split has an empty article.")
                    identity = str(row["meta"]["id"])
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    heldout_ids.add(identity)
                    heldout_text.add(digest)
                    emit(handle, {"text": text, "meta": row["meta"], "text_sha256": digest,
                                  "source": validation[0].name, "source_line": index, "split": "validation"})
                    counts["validation_articles"] += 1
        for part, path in enumerate(training):
            rows = []
            with gzip.open(path, "rt") as source:
                for index, line in enumerate(source):
                    row = json.loads(line)
                    counts["raw_train_articles"] += 1
                    text = row["text"].strip()
                    if not text:
                        counts["empty_train_articles"] += 1
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if str(row["meta"]["id"]) in heldout_ids or digest in heldout_text:
                        counts["validation_overlap_removed"] += 1
                        continue
                    rows.append({"text": text, "meta": row["meta"], "text_sha256": digest,
                                 "source": path.name, "source_line": index, "split": "train"})
            random.Random(16092026 + part).shuffle(rows)
            with (temporary / f"{part:02d}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["train_articles"] += len(rows)
            print(json.dumps({"stage": "wikipedia_shuffled", "file": path.name, "articles": len(rows)}), flush=True)
            del rows
        handles = [path.open() for path in sorted(temporary.iterdir())]
        try:
            with (output / "wiki-train.txt").open("wb") as handle:
                for batch in itertools.zip_longest(*handles):
                    for line in batch:
                        if line is not None:
                            emit(handle, json.loads(line))
        finally:
            for handle in handles:
                handle.close()
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    return dict(counts)


def expanded_recipe(base, data_path):
    recipe = copy.deepcopy(base)
    for language in ("en", "ja"):
        recipe["sources"].append({
            "name": f"aya_{language}", "kind": "conversations",
            "path": f"{data_path}/aya-{language}-train.jsonl",
            "evaluation": {"path": f"{data_path}/aya-{language}-validation.jsonl"},
            "conversation_objective": "assistant", "conversation_tokens": 2048, "conversation_overlap": 1024,
            "records_per_update": 4, "weight": 1.0, "question_mode": "fixed", "question_evaluation_examples": 0,
        })
    recipe["sources"].append({
        "name": "wikipedia_ja", "kind": "text", "path": f"{data_path}/wiki-train.txt",
        "evaluation": {"path": f"{data_path}/wiki-validation.txt"},
        "block_tokens": 512, "records_per_update": 1, "weight": 1.0,
        "question_mode": "fixed", "question_evaluation_examples": 0,
    })
    return recipe


def main():
    import pyarrow.parquet as parquet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--evaluation-prompts", type=Path, action="append", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    forbidden = {prompt_key(row["prompt"]) for path in args.evaluation_prompts for row in json.loads(path.read_text())}
    base = json.loads(args.recipe.read_text())
    existing_training = set()
    for source in base["sources"]:
        if source["kind"] != "conversations":
            continue
        for path, target in ((Path(source["path"]), existing_training),
                             (Path(source["evaluation"]["path"]), forbidden)):
            with path.open() as handle:
                for line in handle:
                    target.update(prompt_key(message["content"]) for message in json.loads(line)["messages"]
                                  if message["role"] == "user")
    aya_manifest = json.loads((args.raw / "aya-source.json").read_text())
    aya_file = args.raw / "aya-train.parquet"
    if file_identity(aya_file)["sha256"] != aya_manifest["sha256"]:
        raise ValueError("Aya input does not match its pinned source.")
    populations, counts = aya_rows(parquet.read_table(aya_file).to_pylist(), forbidden, existing_training)
    for (language, split), rows in populations.items():
        if not rows:
            raise ValueError("Every language must have a training and validation population.")
        with (args.output / f"aya-{language}-{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    wiki_manifest = json.loads((args.raw / "wiki-downloads.json").read_text())
    wiki_counts = prepare_wikipedia(args.raw, args.output, wiki_manifest)
    write_json(args.output / "existing-recipe.json", base)
    write_json(args.output / "expanded-recipe.json", expanded_recipe(base, args.output.as_posix()))
    write_json(args.output / "provenance.json", {
        "schema": "intrep.replay-coverage-data.v1", "aya": aya_manifest, "aya_counts": counts,
        "aya_populations": {f"{language}_{split}": len(rows) for (language, split), rows in populations.items()},
        "aya_split": "Normalized prompt SHA-256 modulo 20; bucket zero is validation unless that prompt already appears in existing training. Official test is not used.",
        "wikipedia": wiki_manifest, "wikipedia_counts": wiki_counts,
        "wikipedia_order": "Seeded shuffle within every upstream shard, followed by article round-robin across all shards.",
        "wikipedia_validation": "Published validation; matching page IDs and exact article texts excluded from training.",
        "evaluation_files": [file_identity(path) for path in args.evaluation_prompts],
        "files": {path.name: file_identity(path) for path in sorted(args.output.iterdir()) if path.is_file()},
        "limits": "These are alternative replay data, not Liquid AI's original training corpus. No general capability guarantee.",
    })
    print(json.dumps({"stage": "prepared", "aya": counts, "wikipedia": wiki_counts}), flush=True)


if __name__ == "__main__":
    main()
