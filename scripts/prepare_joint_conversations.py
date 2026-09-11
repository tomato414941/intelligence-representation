"""Export all available OASST1 conversation branches, with no size or language cap."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from intrep.problems.shared_prediction.streams import file_identity

REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"


def conversation_branches(messages):
    chains = {}
    for identifier, row in messages.items():
        chain, visited, current = [], set(), row
        while current is not None:
            key = current["message_id"]
            if key in visited:
                raise ValueError("conversation contains a parent cycle")
            visited.add(key)
            if current.get("deleted") or not isinstance(current.get("text"), str) or not current["text"].strip():
                chain = []
                break
            chain.append(current)
            parent = current.get("parent_id")
            if parent is not None and parent not in messages:
                raise ValueError("conversation parent is absent from the archive")
            current = messages.get(parent)
        if chain:
            chain.reverse()
            if chain[0]["role"] != "prompter" or any(item["message_tree_id"] != row["message_tree_id"] for item in chain):
                raise ValueError("conversation has an invalid root or tree identity")
            chains[identifier] = chain
    parents = {messages[key].get("parent_id") for key in chains}
    # Ancestors appear in each complete branch; every usable message is covered.
    return [chain for key, chain in chains.items() if key not in parents], len(chains)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.archive, "rt", encoding="utf-8") as handle:
        rows = list(map(json.loads, handle))
    messages = {row["message_id"]: row for row in rows}
    if len(messages) != len(rows):
        raise ValueError("archive contains duplicate message identities")
    branches, usable = conversation_branches(messages)
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "validation": 0}
    handles = {split: (args.output / f"{split}.jsonl").open("w", encoding="utf-8") for split in counts}
    covered = {split: set() for split in counts}
    try:
        for chain in branches:
            row = chain[-1]
            group = row["message_tree_id"]
            split = "validation" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
            output = {"id": row["message_id"], "group_id": group,
                      "source": f"OpenAssistant/oasst1@{REVISION}",
                      "messages": [{"role": "user" if item["role"] == "prompter" else "assistant",
                                    "content": item["text"]} for item in chain]}
            handles[split].write(json.dumps(output, ensure_ascii=False) + "\n")
            counts[split] += 1
            covered[split].update(item["message_id"] for item in chain)
    finally:
        for handle in handles.values():
            handle.close()
    assert not covered["train"] & covered["validation"]
    assert sum(map(len, covered.values())) == usable
    shutil.copyfile(args.archive.parent / "LICENSE", args.output / "LICENSE")
    provenance = {"dataset": "OpenAssistant/oasst1", "revision": REVISION, "license": "Apache-2.0",
                  "archive": file_identity(args.archive), "archive_messages": len(messages),
                  "usable_messages": usable, "excluded_messages": len(messages) - usable,
                  "branches": counts, "messages_by_split": {split: len(ids) for split, ids in covered.items()},
                  "selection": "Every root-to-leaf branch; all languages, ranks and lengths. Exclude deleted/empty messages and their descendants. Split by tree hash."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
