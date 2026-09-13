"""Temporary, auditable lessons for the text-to-image order experiment."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from intrep.problems.shared_prediction.answers import question_prefix
from intrep.problems.shared_prediction.rule_transfer import (
    MeasuredReadout, accuracy, digit_question, image_record, order_question,
)
from intrep.problems.shared_prediction.rule_transfer_data import DIGITS, precedes, text_training_examples, validate_image_manifest
from intrep.problems.shared_prediction.sources import source_configs
from intrep.problems.shared_prediction.streams import EpochSampler

LESSON_KEY = "rule_transfer_lessons"
LESSON_NAMES = ("digit_names", "old_image_order", "old_text_order", "text_tuition")
LESSON_SLOTS = {**{name: index for index, name in enumerate(LESSON_NAMES)}, "image_tuition": 3}


def state_digest(value):
    """Hash values rather than serialization container IDs or object addresses."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")

    visit(value)
    return digest.hexdigest()


def training_configs(recipe):
    return [{key: value for key, value in row.items() if key != "evaluation"}
            for row in source_configs(recipe)]


def validate_training_recipe(recipe, previous, panel, root, *, required_sources=12):
    if len(recipe["sources"]) != required_sources or training_configs(recipe) != training_configs(previous):
        raise ValueError("preserve all original training source configurations and populations")
    mnist = next(row for row in recipe["sources"] if row["name"] == "mnist")
    if (root / mnist["images"]).resolve() != (root / panel["files"]["training_images"]["path"]).resolve():
        raise ValueError("supplemental grounding must use the audited MNIST training images")
    excluded = set(mnist["evaluation"].get("evaluation_excluded_indices", []))
    if not set(panel["holdout_excluded_indices"]) <= excluded:
        raise ValueError("ordinary evaluation must exclude every reserved holdout image")


def batched_answer_loss(source, questions):
    """Match the existing answer loss while batching equal-length questions."""
    if not questions:
        raise ValueError("a lesson must contain questions")
    groups = defaultdict(list)
    for question in questions:
        if question.answer is None or question.predictions:
            raise ValueError("these lessons require explicit text answer supervision")
        observations = [source.model.encode(name, *values) for name, values in question.inputs]
        prefix = question_prefix(source, question.prompt, observations)
        targets = source.text_ids(question.answer) + [source.tokenizer.eos_token_id]
        groups[prefix.shape[1], len(targets)].append((prefix, targets))
    losses = []
    for (prefix_length, _), rows in groups.items():
        prefix = torch.cat([row[0] for row in rows], dim=0)
        targets = torch.tensor([row[1] for row in rows], device=source.device, dtype=torch.long)
        suffix = source.model.encode("text", targets[:, :-1])
        hidden = source.model(torch.cat((prefix, suffix), dim=1))[:, prefix_length - 1:]
        logits = source.model.decode("text", hidden)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        losses.append(loss * len(rows) / len(questions))
    return sum(losses)


class RuleLessons:
    """Keep supplemental sampling separate from all twelve background readers."""

    def __init__(self, source, orders, *, condition="calibration", seed=47, batches=(16, 8, 8, 8),
                 manifest=None, image_manifest=None):
        if condition not in ("calibration", "a", "b", "control") or len(batches) != 4 or min(batches) < 1:
            raise ValueError("choose a lesson condition and four positive batch sizes")
        self.source, self.orders, self.condition = source, copy.deepcopy(orders), condition
        self.batches, self.seed = tuple(batches), seed
        self.pairs = [(a, b) for a in DIGITS for b in DIGITS if a != b]
        self.class_indices = {digit: np.flatnonzero(np.asarray(source.labels) == digit).tolist() for digit in DIGITS}
        if any(not indices for indices in self.class_indices.values()):
            raise ValueError("grounding needs the complete ten-digit training population")
        self.samplers = {
            "names": EpochSampler(len(source.labels), seed + 101),
            "image_pairs": EpochSampler(90, seed + 102),
            "old_text_pairs": EpochSampler(90, seed + 103),
            **{f"class_{digit}": EpochSampler(len(indices), seed + 200 + digit)
               for digit, indices in self.class_indices.items()},
        }
        self.tuition_position = 0
        self.manifest = None
        self.image_manifest = copy.deepcopy(image_manifest)
        if image_manifest is not None:
            if condition not in ("a", "control") or manifest is not None:
                raise ValueError("image followup compares A/control without ongoing new text tuition")
            validate_image_manifest(image_manifest, source.images, source.labels, orders)
        elif condition != "calibration":
            expected = {"schema_version": "intrep.rule_transfer_text.v1", "condition": condition,
                        "examples": text_training_examples(orders, condition)}
            if manifest != expected:
                raise ValueError("text tuition must exactly match the prepared condition manifest")
            self.manifest = copy.deepcopy(manifest)
        self.last_trace = {}

    @property
    def names(self):
        if self.image_manifest is not None:
            return (*LESSON_NAMES[:3], "image_tuition")
        return LESSON_NAMES[:3] if self.condition == "calibration" else LESSON_NAMES

    def state_dict(self):
        return {"samplers": {name: sampler.state_dict() for name, sampler in self.samplers.items()},
                "tuition_position": self.tuition_position}

    def load_state_dict(self, state):
        if set(state["samplers"]) != set(self.samplers) or state["tuition_position"] < 0:
            raise ValueError("lesson sampling state differs")
        for name, sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][name])
        self.tuition_position = state["tuition_position"]

    def provenance(self):
        result = {"condition": self.condition, "batches": list(self.batches), "seed": self.seed,
                  "orders": self.orders, "manifest": self.manifest,
                  "image_supervision": "MNIST training split; digit naming and old order only",
                  "new_rule_supervision": "text observations only"}
        if self.image_manifest is not None:
            result.update(image_manifest=self.image_manifest,
                          image_supervision="MNIST training split; digit naming, old order and the fixed new-order support",
                          new_rule_supervision="image observations only in this followup phase")
        return result

    def questions(self, name):
        if name not in self.names:
            raise ValueError("lesson is not active in this condition")
        questions, trace = [], []
        for _ in range(self.batches[LESSON_SLOTS[name]]):
            if name == "digit_names":
                index = self.samplers["names"].next()
                questions.append(digit_question(self.source.read_record(index)))
                trace.append(index)
            elif name == "old_image_order":
                digits = self.pairs[self.samplers["image_pairs"].next()]
                indices = [self.class_indices[digit][self.samplers[f"class_{digit}"].next()] for digit in digits]
                records = [self.source.read_record(index) for index in indices]
                questions.append(order_question(self.source, records, rule="old", order=self.orders["old"]))
                trace.append(indices)
            elif name == "old_text_order":
                digits = self.pairs[self.samplers["old_text_pairs"].next()]
                questions.append(order_question(self.source, digits, rule="old", modality="text", order=self.orders["old"]))
                trace.append(digits)
            elif name == "image_tuition":
                examples = self.image_manifest["examples"]
                row = examples[self.tuition_position % len(examples)]
                self.tuition_position += 1
                question = order_question(self.source, [self.source.read_record(index) for index in row["indices"]])
                question.answer = row["answer"]
                questions.append(question)
                trace.append(row["id"])
            else:
                row = self.manifest["examples"][self.tuition_position % 90]
                self.tuition_position += 1
                question = order_question(self.source, row["digits"], rule=row["rule"], modality="text")
                question.answer = row["answer"]
                questions.append(question)
                trace.append(row["id"])
        self.last_trace[name] = trace
        return questions

    def loss(self, name):
        return batched_answer_loss(self.source, self.questions(name))


def measure_prerequisites(source, panel, images, labels, *, condition, readout=None, progress=None):
    """Inspect development prerequisites without querying new-rule image answers."""
    if condition not in ("calibration", "a", "b", "control"):
        raise ValueError("unknown calibration/intervention condition")
    selected = panel["panels"]["development"]
    readout = readout or MeasuredReadout(source)
    mode = source.model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if source.device.type == "cuda" else []
    source.model.eval()
    digits, old_images, old_text, new_text = [], [], [], []
    try:
        for index in sorted({index for pair in selected for index in pair["indices"]}):
            response = readout(digit_question(image_record(source, images[index])), tuple(map(str, DIGITS)))
            digits.append({"index": index, "expected": str(int(labels[index])), **response})
            if progress and len(digits) % 100 == 0:
                progress({"stage": "prerequisite_digits", "complete": len(digits), "total": 2 * len(selected)})
        for pair in selected:
            for orientation in (0, 1):
                indices = pair["indices"][::1 if orientation == 0 else -1]
                observations = [image_record(source, images[index]) for index in indices]
                response = readout(order_question(source, observations, rule="old"), ("yes", "no"))
                expected = "yes" if int(labels[indices[0]]) < int(labels[indices[1]]) else "no"
                old_images.append({"pair_id": pair["id"], "indices": indices, "expected": expected, **response})
            if progress and len(old_images) % 100 == 0:
                progress({"stage": "prerequisite_old_images", "complete": len(old_images), "total": 2 * len(selected)})
        for target, rule_condition in ((old_text, "control"), (new_text, condition)):
            if target is new_text and condition in ("calibration", "control"):
                continue
            for row in text_training_examples(panel["orders"], rule_condition):
                response = readout(order_question(source, row["digits"], rule=row["rule"], modality="text"), ("yes", "no"))
                target.append({"digits": row["digits"], "expected": row["answer"], **response})
    finally:
        source.model.train(mode)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    sets = {"digit_naming": (digits, .95), "old_image_rule": (old_images, .90), "old_text_rule": (old_text, .95)}
    if new_text:
        sets["new_text_rule"] = (new_text, .95)
    gates = {name: {"accuracy": accuracy(rows), "count": len(rows), "threshold": threshold,
                    "passed": accuracy(rows) >= threshold} for name, (rows, threshold) in sets.items()}
    return {"schema_version": "intrep.rule_transfer_prerequisites.v1", "split": "development", "condition": condition,
            "gates": gates, "passed": all(row["passed"] for row in gates.values()),
            "digit_readouts": digits, "old_image_rows": old_images, "old_text_rows": old_text, "new_text_rows": new_text,
            "costs": readout.snapshot(), "new_rule_image_queries": 0}


def measure_image_followup(source, panel, images, labels, lessons, *, condition, readout=None, progress=None):
    """Measure development transfer and training fit with explicit image-query counts."""
    readout = readout or MeasuredReadout(source)
    mode, cpu_rng = source.model.training, torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if source.device.type == "cuda" else []
    source.model.eval()
    try:
        result = measure_prerequisites(source, panel, images, labels, condition=condition,
                                       readout=readout, progress=progress)
        new_rows, support_rows = [], []
        for pair_number, pair in enumerate(panel["panels"]["development"], 1):
            for orientation, direction in ((0, 1), (1, -1)):
                indices = pair["indices"][::direction]
                response = readout(order_question(source, [image_record(source, images[index]) for index in indices]), ("yes", "no"))
                expected = "yes" if precedes(panel["orders"]["a"], *[int(labels[index]) for index in indices]) else "no"
                new_rows.append({"pair_id": pair["id"], "orientation": orientation, "indices": indices,
                                 "class_pair": pair["digits"], "expected": expected, **response})
            if progress and pair_number % 50 == 0:
                progress({"stage": "followup_development_images", "complete": pair_number,
                          "total": len(panel["panels"]["development"])})
        for row in lessons.image_manifest["examples"]:
            response = readout(order_question(source, [lessons.source.read_record(index) for index in row["indices"]]), ("yes", "no"))
            support_rows.append({"id": row["id"], "training_indices": row["indices"], "expected": row["answer"], **response})
        score = accuracy(new_rows)
        result["gates"]["new_image_rule"] = {"accuracy": score, "count": len(new_rows), "threshold": .90, "passed": score >= .90}
        result.update(schema_version="intrep.rule_transfer_image_prerequisites.v1",
                      passed=all(gate["passed"] for gate in result["gates"].values()),
                      new_image_rows=new_rows, support_rows=support_rows,
                      support_accuracy=accuracy(support_rows), costs=readout.snapshot(),
                      new_rule_image_queries=len(new_rows) + len(support_rows),
                      development_new_rule_image_queries=len(new_rows), support_new_rule_image_queries=len(support_rows))
        return result
    finally:
        source.model.train(mode)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
