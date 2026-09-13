"""Question construction and measured readouts for the order-transfer experiment."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch

from intrep.problems.shared_prediction.answers import question_prefix
from intrep.problems.shared_prediction.questions import Question
from intrep.problems.shared_prediction.rule_transfer_data import DIGITS, precedes


def image_record(source, pixels, index=None, label=None):
    rgb = np.repeat(np.asarray(pixels)[:, :, None], 3, axis=-1)
    return {"image": torch.tensor(rgb.copy(), device=source.device, dtype=source.dtype) / 255,
            "index": index, "label": label}


def digit_question(record):
    return Question("Name the digit in the observation. Return only the digit.",
                    answer=None if record["label"] is None else str(record["label"]),
                    inputs=[("rgb", (record["image"],))])


def order_question(source, observations, *, rule="new", modality="image", order=None):
    if rule not in ("old", "new") or modality not in ("image", "text") or len(observations) != 2:
        raise ValueError("an order question needs two observations and an explicit old/new rule")
    inputs = []
    for number, observation in enumerate(observations, 1):
        marker = f"\nObservation {number}:\n"
        if modality == "text":
            if type(observation) is not int or observation not in DIGITS:
                raise ValueError("text observations must be digit IDs")
            inputs.append(("text", (source.ids(source.text_ids(marker + str(observation))),)))
        else:
            inputs.append(("text", (source.ids(source.text_ids(marker)),)))
            inputs.append(("rgb", (observation["image"],)))
    digits = observations if modality == "text" else [row["label"] for row in observations]
    answer = None if order is None else "yes" if precedes(order, *digits) else "no"
    return Question(f"Under the {rule} order, does observation 1 come before observation 2? Answer only yes or no.",
                    answer, inputs)


class MeasuredReadout:
    """Greedy text answers plus candidate probabilities from the same first logits."""

    def __init__(self, source, max_tokens=8):
        if max_tokens < 1:
            raise ValueError("generation needs a positive token budget")
        self.source, self.max_tokens = source, max_tokens
        self.calls = self.positions = 0
        self.seconds = 0.0

    def snapshot(self):
        return {"core_calls": self.calls, "core_input_positions": self.positions, "seconds": self.seconds}

    @torch.inference_mode()
    def __call__(self, question, candidates=()):
        source = self.source
        candidate_ids = [source.text_ids(value) for value in candidates]
        if any(len(ids) != 1 for ids in candidate_ids) or len({ids[0] for ids in candidate_ids}) != len(candidates):
            raise ValueError("the experiment requires distinct single-token candidate strings")
        if source.device.type == "cuda":
            torch.cuda.synchronize(source.device)
        start = time.perf_counter()
        observations = [source.model.encode(name, *values) for name, values in question.inputs]
        context = question_prefix(source, question.prompt, observations)
        initial_positions = context.shape[1]
        generated, probabilities, candidate_mass = [], {}, None
        for step in range(self.max_tokens):
            self.calls += 1
            self.positions += context.shape[1]
            hidden = source.model(context)[:, -1:]
            logits = source.model.decode("text", hidden)[0, 0].float()
            if step == 0 and candidates:
                ids = [value[0] for value in candidate_ids]
                probabilities = dict(zip(candidates, logits[ids].softmax(-1).tolist()))
                candidate_mass = float(logits.softmax(-1)[ids].sum())
            token = int(logits.argmax())
            if token == source.tokenizer.eos_token_id:
                break
            generated.append(token)
            context = torch.cat((context, source.model.encode("text", source.ids([token]))), dim=1)
        if source.device.type == "cuda":
            torch.cuda.synchronize(source.device)
        elapsed = time.perf_counter() - start
        self.seconds += elapsed
        return {"answer": source.tokenizer.decode(generated, skip_special_tokens=True),
                "probabilities": probabilities, "candidate_mass": candidate_mass,
                "generated_tokens": len(generated), "prefix_tokens": initial_positions, "seconds": elapsed}


def difference(after, before):
    return {key: after[key] - before[key] for key in after}


def strict_digit(text):
    text = text.strip()
    return int(text) if text in tuple(map(str, DIGITS)) else None


def accuracy(rows, field="answer"):
    return sum(row[field].strip() == row["expected"] for row in rows) / len(rows) if rows else None


def summarize_transfer(rows):
    result = {}
    by_route = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    for route, cases in by_route.items():
        groups = defaultdict(list)
        for case in cases:
            groups[case["pair_id"]].append(case)
        if any(len(group) != 2 or {row["orientation"] for row in group} != {0, 1} for group in groups.values()):
            raise ValueError("each route must include both orientations of every pair")
        result[route] = {"answers": len(cases), "accuracy": accuracy(cases), "pairs": len(groups),
                         "both_orientations_correct": sum(all(row["answer"].strip() == row["expected"] for row in group)
                                                          for group in groups.values()) / len(groups),
                         "changed_accuracy": accuracy([row for row in cases if row["changed"]]),
                         "unchanged_accuracy": accuracy([row for row in cases if not row["changed"]])}
    return result


def compare_counterfactuals(first, second):
    """Require both orientations and both independently taught rule branches."""
    if first["panel_sha256"] != second["panel_sha256"] or first["split"] != second["split"]:
        raise ValueError("counterfactual reports must evaluate the same fixed panel and split")
    if {first["order"], second["order"]} != {"a", "b"}:
        raise ValueError("counterfactual comparison needs orders a and b")
    results = {}
    for route in ("direct", "read_then_apply", "cached_posterior", "oracle_digits"):
        a = {(row["pair_id"], row["orientation"]): row for row in first["rows"] if row["route"] == route}
        b = {(row["pair_id"], row["orientation"]): row for row in second["rows"] if row["route"] == route}
        if not a or set(a) != set(b):
            raise ValueError("counterfactual routes contain different cases")
        groups = defaultdict(list)
        for key, left in a.items():
            right = b[key]
            if left["changed"] != right["changed"] or (left["expected"] != right["expected"]) != left["changed"]:
                raise ValueError("counterfactual labels disagree with their declared strata")
            groups[left["pair_id"]].extend((left, right))
        if any(len(rows) != 4 for rows in groups.values()):
            raise ValueError("a counterfactual group requires all four answers")
        results[route] = {}
        for stratum in (True, False):
            selected = [rows for rows in groups.values() if rows[0]["changed"] == stratum]
            results[route]["changed" if stratum else "unchanged"] = {
                "pairs": len(selected),
                "all_four_correct": sum(all(row["answer"].strip() == row["expected"] for row in rows)
                                        for rows in selected) / len(selected) if selected else None,
            }
    return results


def evaluate_transfer(source, panel, images, labels, *, split, order_name, readout=None, progress=None):
    """Evaluate only the selected split; never send expected answers to the readout."""
    if split not in ("development", "holdout") or order_name not in ("a", "b"):
        raise ValueError("choose a development/holdout split and counterfactual order a/b")
    selected = panel["panels"][split]
    order = panel["orders"][order_name]
    readout = readout or MeasuredReadout(source)
    costs = {}
    previous_mode = source.model.training
    source.model.eval()
    try:
        start = readout.snapshot()
        digit_rows, digit_cache = [], {}
        for index in sorted({index for row in selected for index in row["indices"]}):
            record = image_record(source, images[index])
            response = readout(digit_question(record), tuple(map(str, DIGITS)))
            digit_cache[index] = response
            digit_rows.append({"index": index, "expected": str(int(labels[index])), **response})
            if progress and len(digit_rows) % 100 == 0:
                progress({"stage": "digit_readout", "complete": len(digit_rows), "total": 2 * len(selected)})
        costs["digit_cache"] = difference(readout.snapshot(), start)

        # The table is the model's own answers/probabilities, including its errors.
        # True image labels are used only by the explicitly named oracle route.
        start = readout.snapshot()
        table, text_rows = {}, []
        for a in DIGITS:
            for b in DIGITS:
                response = readout(order_question(source, [a, b], modality="text"), ("yes", "no"))
                table[a, b] = response
                text_rows.append({"digits": [a, b], "expected": "yes" if precedes(order, a, b) else "no", **response})
        costs["text_rule_cache"] = difference(readout.snapshot(), start)

        rows, old_rows = [], []
        direct_before = readout.snapshot()
        old_cost = {"core_calls": 0, "core_input_positions": 0, "seconds": 0.0}
        composition_seconds = 0.0
        for pair_number, pair in enumerate(selected, 1):
            for orientation in (0, 1):
                indices = pair["indices"][::1 if orientation == 0 else -1]
                digits = [int(labels[index]) for index in indices]
                records = [image_record(source, images[index]) for index in indices]
                common = {"pair_id": pair["id"], "orientation": orientation, "indices": indices,
                          "class_pair": pair["digits"], "changed": pair["changed"],
                          "expected": "yes" if precedes(order, *digits) else "no"}
                direct = readout(order_question(source, records), ("yes", "no"))
                rows.append({**common, "route": "direct", **direct})

                before = readout.snapshot()
                old = readout(order_question(source, records, rule="old"), ("yes", "no"))
                for key, value in difference(readout.snapshot(), before).items():
                    old_cost[key] += value
                old_rows.append({**common, "expected": "yes" if precedes(panel["orders"]["old"], *digits) else "no", **old})

                start_composition = time.perf_counter()
                inferred = [strict_digit(digit_cache[index]["answer"]) for index in indices]
                hard = table[tuple(inferred)]["answer"] if None not in inferred else ""
                rows.append({**common, "route": "read_then_apply", "answer": hard, "read_digits": inferred})
                probabilities = [digit_cache[index]["probabilities"] for index in indices]
                posterior = sum(probabilities[0][str(a)] * probabilities[1][str(b)] * table[a, b]["probabilities"]["yes"]
                                for a in DIGITS for b in DIGITS)
                rows.append({**common, "route": "cached_posterior", "answer": "yes" if posterior > 0.5 else "no",
                             "yes_probability": posterior})
                rows.append({**common, "route": "oracle_digits", "answer": table[tuple(digits)]["answer"]})
                composition_seconds += time.perf_counter() - start_composition
            if progress and pair_number % 25 == 0:
                progress({"stage": "image_relations", "complete": pair_number, "total": len(selected)})
        elapsed = difference(readout.snapshot(), direct_before)
        costs["direct"] = {key: elapsed[key] - old_cost[key] for key in elapsed}
        costs["old_rule_control"] = old_cost
        costs["cached_composition_seconds"] = composition_seconds
        gates = {"digit_naming": {"accuracy": accuracy(digit_rows), "threshold": 0.95},
                 "new_rule_text": {"accuracy": accuracy([row for row in text_rows if row["digits"][0] != row["digits"][1]]),
                                   "threshold": 0.95},
                 "old_rule_images": {"accuracy": accuracy(old_rows), "threshold": 0.90}}
        for gate in gates.values():
            gate["passed"] = gate["accuracy"] >= gate["threshold"]
        return {"schema_version": "intrep.rule_transfer_evaluation.v1", "split": split, "order": order_name,
                "gates": gates, "transfer_interpretable": all(gate["passed"] for gate in gates.values()),
                "summary": summarize_transfer(rows), "rows": rows, "digit_readouts": digit_rows,
                "text_rule_readouts": text_rows, "old_rule_readouts": old_rows, "costs": costs,
                "cost_scope": "Synchronized model inference, including input/output layers and greedy decoding; image file loading is excluded. Core input positions are not FLOPs. Cache construction is reported separately and must be counted before reuse.",
                "baseline_scope": "Digit probabilities and the 100-entry rule table come from this same checkpoint. No true image class or true order table is supplied to the two learned bridge routes. Oracle digits are diagnostic only.",
                "limitations": "Transfer of a newly taught order through supervised digit grounding; not unsupervised grounding, an unseen operator family, or evidence of superiority to symbolic representations."}
    finally:
        source.model.train(previous_mode)
