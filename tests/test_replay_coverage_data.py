from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from intrep.problems.shared_prediction.recipe import validate_extension
from intrep.problems.shared_prediction.streams import file_identity
from scripts.prepare_replay_coverage import aya_rows, expanded_recipe, prepare_wikipedia, prompt_group, prompt_key


class ReplayCoverageTests(unittest.TestCase):
    def test_prompt_groups_cannot_cross_splits_or_reuse_evaluation_prompts(self):
        prompt = next(f"example {index}" for index in range(1000)
                      if int(prompt_group(f"example {index}")[:16], 16) % 20 == 0)
        rows = [{"inputs": value, "targets": f"answer {index}", "language_code": language,
                 "annotation_type": "original-annotations"}
                for index, (value, language) in enumerate([
                    (prompt, "eng"), (prompt + "  \n", "jpn"), ("forbidden", "eng"), ("", "jpn"),
                ])]
        populations, counts = aya_rows(rows, {"forbidden"})
        self.assertEqual(sum(len(rows) for rows in populations.values()), 2)
        self.assertEqual(len(populations["en", "validation"]), 1)
        self.assertEqual(len(populations["ja", "validation"]), 1)
        self.assertEqual(counts["evaluation_prompt_overlap"], 1)
        moved, counts = aya_rows(rows, {"forbidden"}, {prompt_key(prompt)})
        self.assertEqual(len(moved["en", "train"]), 1)
        self.assertEqual(len(moved["ja", "train"]), 1)
        self.assertEqual(counts["validation_prompt_already_in_existing_training"], 2)

    def test_extension_preserves_every_existing_source_and_setting(self):
        base = {"seed": 47, "defaults": {"weight": 1.0}, "sources": [
            {"name": "previous", "kind": "text", "path": "train.txt", "evaluation": {"path": "valid.txt"}},
        ]}
        result = expanded_recipe(base, "data/example")
        validate_extension(base, result)
        self.assertEqual(len(base["sources"]), 1)
        self.assertEqual(len(result["sources"]), 4)

    def test_all_wikipedia_shards_survive_with_validation_pages_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw", root / "out"
            raw.mkdir()
            output.mkdir()
            files = []
            for part in range(15):
                path = raw / (f"train_{part}.jsonl.gz" if part < 14 else "validation_0.jsonl.gz")
                rows = [{"text": f"article {part}\nsecond paragraph", "meta": {"id": str(part)}}]
                if part == 0:
                    rows.extend([{"text": "article 14\nsecond paragraph", "meta": {"id": "different"}},
                                 {"text": "older heldout version", "meta": {"id": "14"}}])
                with gzip.open(path, "wt") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                files.append({"path": path.name, **file_identity(path)})
            counts = prepare_wikipedia(raw, output, {"files": files})
            self.assertEqual(counts["train_articles"], 14)
            self.assertEqual(counts["validation_overlap_removed"], 2)
            text = (output / "wiki-train.txt").read_text()
            for part in range(14):
                self.assertEqual(text.count(f"article {part}\n"), 1)
            self.assertNotIn("article 14\n", text)
            self.assertNotIn("older heldout version", text)
            with gzip.open(output / "wiki-articles.jsonl.gz", "rt") as handle:
                index = [json.loads(line) for line in handle]
            for row in index:
                path = output / f"wiki-{row['split']}.txt"
                fragment = path.read_bytes()[row["start"]:row["end"]]
                self.assertTrue(fragment.endswith(b"\n\n"))
            self.assertFalse((output / ".wiki-shuffle").exists())


if __name__ == "__main__":
    unittest.main()
