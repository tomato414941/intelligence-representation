from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.problems.shared_prediction.rule_transfer_data import file_digest
from scripts.run_rule_transfer_interventions import main


class InterventionScheduleTests(unittest.TestCase):
    def exercise(self, root, *, b_passes):
        common, panel, output, work = [root / name for name in ("common", "panel", "output", "work")]
        common.mkdir()
        panel.mkdir()
        (common / "checkpoint.pt").write_bytes(b"synthetic scheduling fixture")
        (panel / "panel.json").write_text("{}")
        (common / "result.json").write_text(json.dumps({
            "condition": "calibration", "prerequisites_passed": True,
            "checkpoint_sha256": file_digest(common / "checkpoint.pt"),
            "settings": {"panel_sha256": file_digest(panel / "panel.json"), "seed": 47, "learning_rate": 1e-5,
                         "batches": [16, 8, 8, 8], "weights": [8, 8, 2, 8]},
        }))
        calls = []

        def run(name, arguments):
            def value(flag):
                return arguments[arguments.index(flag) + 1]

            calls.append((name, arguments))
            if name == "train_rule_transfer.py":
                condition, steps = value("--condition"), int(value("--steps"))
                passed = condition != "b" or b_passes and steps >= 450
                directory = Path(value("--output"))
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "checkpoint.pt").write_bytes(b"synthetic branch")
                (directory / "result.json").write_text(json.dumps({"prerequisites_passed": passed,
                    "prerequisites": {"fixture_gate": {"passed": passed}}, "completed_steps": steps}))
            elif name in ("audit_rule_transfer_training.py", "evaluate_rule_transfer.py", "compare_rule_transfer.py"):
                Path(value("--output")).write_text("{}")

        argv = ["run_rule_transfer_interventions.py", "--common", str(common), "--panel-directory", str(panel),
                "--work", str(work), "--output", str(output), "--archive-prefix", "fixture/archive", "--milestones", "225", "450"]
        with patch("sys.argv", argv), patch("scripts.run_rule_transfer_interventions.script", side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        return calls, json.loads((output / "outcome.json").read_text())

    def test_both_branches_and_control_use_a_common_passing_update_count(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, outcome = self.exercise(Path(directory), b_passes=True)
            training = [(args[args.index("--condition") + 1], args[args.index("--steps") + 1], args[0])
                        for name, args in calls if name == "train_rule_transfer.py"]
            self.assertEqual(training, [("a", 225, "--common"), ("b", 225, "--common"),
                                        ("a", 450, "--resume"), ("b", 450, "--resume"), ("control", 450, "--common")])
            first_evaluation = next(index for index, (name, _) in enumerate(calls) if name == "evaluate_rule_transfer.py")
            self.assertTrue(all(name != "train_rule_transfer.py" for name, _ in calls[first_evaluation:]))
            self.assertEqual(calls[first_evaluation - 1][0], "audit_rule_transfer_training.py")
            self.assertEqual(outcome["selected_updates"], 450)
            self.assertTrue(outcome["development_transfer_evaluated"])
            self.assertFalse(outcome["holdout_evaluated"])
            self.assertTrue(all(args[args.index("--split") + 1] == "development"
                                for name, args in calls if name == "evaluate_rule_transfer.py"))

    def test_unmet_prerequisites_do_not_trigger_transfer_or_holdout_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, outcome = self.exercise(Path(directory), b_passes=False)
            self.assertIsNone(outcome["selected_updates"])
            self.assertFalse(outcome["development_transfer_evaluated"])
            self.assertFalse(outcome["holdout_evaluated"])
            self.assertEqual(outcome["trained_conditions"], ["a", "b"])
            self.assertFalse(any(name == "evaluate_rule_transfer.py" for name, _ in calls))
            self.assertEqual(sum(name == "archive_rule_transfer.py" for name, _ in calls), 2)


if __name__ == "__main__":
    unittest.main()
