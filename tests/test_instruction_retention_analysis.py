from __future__ import annotations

import copy
import unittest

from scripts.summarize_instruction_retention import instruction_scores


class InstructionRetentionAnalysisTests(unittest.TestCase):
    def test_retained_lost_and_newly_correct_are_distinct(self):
        def report(answers):
            return {"generations": [
                {"split": "development", "prompt": prompt, "expected": target,
                 "answer": answer, "exact_match": answer.strip() == target}
                for prompt, target, answer in zip(("英語で答えて", "Calculate", "Format"), ("赤", "17", "AB"), answers)
            ]}
        before = report(["赤", "17", "ab"])
        after = report(["赤", "The answer is 17.", "AB"])
        score = instruction_scores(after, "development", before)
        self.assertEqual((score["correct"], score["questions"]), (2, 3))
        self.assertEqual(score["paired"], {"initially_correct": 2, "retained_correct": 1, "lost_correct": 1, "newly_correct": 1})
        self.assertEqual(score["format_diagnostic"]["strict_correct"], 0)
        self.assertEqual(score["format_diagnostic"]["final_numeral_matches"], 1)
        self.assertEqual(score["groups"]["language/ja"], {"correct": 1, "questions": 1})

    def test_split_separation_and_corrupt_scores_or_targets_are_rejected(self):
        before = {"generations": [{"split": "holdout", "prompt": "A", "expected": "B", "answer": "B", "exact_match": True}]}
        self.assertEqual(instruction_scores(before, "development")["questions"], 0)
        broken = copy.deepcopy(before)
        broken["generations"][0]["answer"] = "C"
        with self.assertRaisesRegex(ValueError, "stored exact-match"):
            instruction_scores(broken, "holdout")
        changed = copy.deepcopy(before)
        changed["generations"][0].update(expected="C", answer="C")
        with self.assertRaisesRegex(ValueError, "identical prompts and targets"):
            instruction_scores(changed, "holdout", before)


if __name__ == "__main__":
    unittest.main()
