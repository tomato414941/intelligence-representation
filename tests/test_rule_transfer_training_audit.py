from __future__ import annotations

import copy
import unittest

from intrep.problems.shared_prediction.rule_transfer_data import make_orders, text_training_examples
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_NAMES
from scripts.audit_rule_transfer_training import audit_conditions


def fixture():
    sources = {f"source_{number}": {"samples": 10} for number in range(12)}
    orders = make_orders(47)
    common = {"condition": "calibration", "completed_steps": 1, "prerequisites_passed": True,
              "source_progress": sources, "checkpoint_sha256": "common-file", "final_parameters_sha256": "common-parameters",
              "initial_background_state_sha256": "initial-readers", "parameters": 100,
              "training_seconds": 10., "lessons": {"orders": orders}}
    common_trace = [{"step": 1, "background_state_sha256": "common-readers"}]
    results, traces = {}, {}
    for name in ("a", "b", "control"):
        results[name] = {"condition": name, "completed_steps": 2,
                         "initial_checkpoint_sha256": "common-file", "initial_parameters_sha256": "common-parameters",
                         "initial_background_state_sha256": "common-readers", "source_progress": sources,
                         "settings": {"condition": name, "manifest_sha256": name + "-manifest", "learning_rate": 1e-5,
                                      "batches": [16, 8, 8, 45]},
                         "optimizer_reset_at_fork": True, "parameters": 100, "trainable_parameters": 100,
                         "new_rule_image_training_examples": 0, "new_rule_image_evaluation_queries": 0,
                         "prerequisites_passed": True, "training_seconds": 5.,
                         "lessons": {"orders": orders, "manifest": {"condition": name,
                             "examples": text_training_examples(orders, name)}}}
        traces[name] = [{"step": step, "background_state_sha256": f"readers-{step}",
                         "losses": {key: 1. for key in [*sources, *LESSON_NAMES, "weighted_loss", "grad_norm"]},
                         "lesson_inputs": {key: [step] * size for key, size in zip(LESSON_NAMES, (16, 8, 8, 45))}} for step in (1, 2)]
        for index, row in enumerate(traces[name]):
            row["lesson_inputs"]["text_tuition"] = [case["id"] for case in results[name]["lessons"]["manifest"]["examples"][45 * index:45 * (index + 1)]]
    return common, results, traces, common_trace


class AuditTests(unittest.TestCase):
    def test_verified_training_is_separate_from_passing_capability_gates(self):
        common, results, traces, common_trace = fixture()
        audited = audit_conditions(common, results, traces, common_trace)
        self.assertTrue(audited["verified"])
        self.assertTrue(audited["prerequisites_passed"])
        results["b"]["prerequisites_passed"] = False
        audited = audit_conditions(common, results, traces, common_trace)
        self.assertTrue(audited["verified"])
        self.assertFalse(audited["prerequisites_passed"])

    def test_invalid_controlled_comparisons_are_rejected(self):
        mutations = {
            "different initial model": lambda c, r, t, ct: r["a"].update(initial_parameters_sha256="different"),
            "different initial readers": lambda c, r, t, ct: r["a"].update(initial_background_state_sha256="different"),
            "different update count": lambda c, r, t, ct: r["b"].update(completed_steps=1),
            "different learning rate": lambda c, r, t, ct: r["control"]["settings"].update(learning_rate=.1),
            "frozen parameters": lambda c, r, t, ct: r["b"].update(trainable_parameters=99),
            "new image supervision": lambda c, r, t, ct: r["a"].update(new_rule_image_training_examples=1),
            "new image model selection": lambda c, r, t, ct: r["a"].update(new_rule_image_evaluation_queries=1),
            "omitted update": lambda c, r, t, ct: t["a"].pop(),
            "omitted calibration trace": lambda c, r, t, ct: ct.clear(),
            "different background samples": lambda c, r, t, ct: t["b"][0].update(background_state_sha256="different"),
            "different supplemental samples": lambda c, r, t, ct: t["control"][0]["lesson_inputs"].update(digit_names=[99]),
            "missing background loss": lambda c, r, t, ct: t["a"][0]["losses"].pop("source_0"),
            "nonfinite loss": lambda c, r, t, ct: t["a"][0]["losses"].update(digit_names=float("nan")),
            "wrong text tuition": lambda c, r, t, ct: r["a"]["lessons"]["manifest"]["examples"][0].update(answer="invalid"),
        }
        for description, mutate in mutations.items():
            with self.subTest(description=description):
                values = copy.deepcopy(fixture())
                mutate(*values)
                with self.assertRaises(ValueError):
                    audit_conditions(*values)


if __name__ == "__main__":
    unittest.main()
