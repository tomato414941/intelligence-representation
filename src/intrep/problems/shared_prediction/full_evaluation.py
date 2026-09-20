"""Evaluate complete held-out populations, streaming individual results to disk."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

import torch

from intrep.problems.shared_prediction.population import (
    completed_epochs,
    population_records,
    primary_stream,
    rewind_population,
)
from intrep.problems.shared_prediction.questions import QuestionSource, stable_seed
from intrep.problems.shared_prediction.sources import NativeSource


class MetricTotals:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, metrics):
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError("evaluation produced a nonfinite metric")
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self):
        return {name: {"mean": total / self.counts[name], "count": self.counts[name]}
                for name, total in self.sums.items()}


def full_cases(source, record_case):
    forms = source.forms if isinstance(source, QuestionSource) and source.config["question_mode"] == "varied" else ("original",)
    for form in forms:
        for wording in (0,) if form == "original" else (0, 1):
            yield {**record_case, "key": f"{record_case['key']}/{form}/{wording}",
                   "record_key": record_case["key"], "form": form, "wording": wording,
                   "seed": stable_seed(record_case["key"])}


@torch.no_grad()
def evaluate_full(model, sources, directory: Path, *, generate_answers=True,
                  omit_native=(), omit_observations=False, only_questions=False):
    """Score every primary record and every configured question, including tails."""
    directory.mkdir(parents=True, exist_ok=True)
    states = {name: copy.deepcopy(source.state_dict()) for name, source in sources.items()}
    limits = {name: primary_stream(source).epoch_limit for name, source in sources.items()}
    native = {name: getattr(source, "reader", source) for name, source in sources.items()
              if isinstance(getattr(source, "reader", source), NativeSource)}
    omissions = {name: source.omitted_inputs.copy() for name, source in native.items()}
    previous_mode, torch_rng = model.training, torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []
    model.eval()
    result = {}
    try:
        for source_index, (name, source) in enumerate(sources.items()):
            rewind_population(source)
            if name in native:
                native[name].omitted_inputs = set(omit_native)
            totals, forms, groups = MetricTotals(), {}, set()
            records = rows = 0
            case_digest = hashlib.sha256()
            path = directory / f"{source_index:03d}.jsonl"
            temporary = path.with_suffix(".jsonl.partial")
            print(json.dumps({"stage": "full_evaluation_source", "source": name}), flush=True)
            with temporary.open("w") as handle:
                for record_case, record in population_records(source):
                    records += 1
                    for case in full_cases(source, record_case):
                        if only_questions and case["form"] == "original":
                            continue
                        if isinstance(source, QuestionSource):
                            paired = case["form"] in ("identify", "same", "different", "sum", "greater")
                            source._forced = {**case, "generate": generate_answers, "omit_observations": omit_observations,
                                              "primary_only": not paired}
                            loss = source.loss(record=record)
                        elif source.config["kind"] == "boolq":
                            loss = source.record_loss(record, generate=generate_answers)
                        else:
                            loss = source.record_loss(record)
                        group = getattr(source, "last_group", None) or case["group"]
                        metrics = {"loss": float(loss), **{key: float(value) for key, value in source.last_metrics.items()}}
                        row = {**case, "group": group, "metrics": metrics,
                               "response": copy.deepcopy(getattr(source, "last_response", None))}
                        totals.add(metrics)
                        forms.setdefault(f"{case['form']}/{case['wording']}", MetricTotals()).add(metrics)
                        groups.add(group)
                        case_digest.update((case["key"] + "\n").encode())
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows += 1
                    if records % 1000 == 0:
                        print(json.dumps({"stage": "full_evaluation_progress", "source": name,
                                          "records": records, "questions": rows}), flush=True)
            if completed_epochs(source) != 1 or not rows:
                raise ValueError(f"full evaluation of {name!r} did not complete a nonempty population")
            temporary.replace(path)
            result[name] = {"summary": totals.summary(), "forms": {key: value.summary() for key, value in forms.items()},
                            "records": records, "examples": rows, "groups": len(groups), "complete": True,
                            "population_progress": getattr(source, "reader", source).progress(),
                            "rows_file": path.name, "case_sha256": case_digest.hexdigest()}
            print(json.dumps({"stage": "full_evaluation_source_complete", "source": name,
                              "records": records, "questions": rows}), flush=True)
        return result
    finally:
        for name, source in sources.items():
            source.load_state_dict(states[name])
            primary_stream(source).epoch_limit = limits[name]
            if name in omissions:
                native[name].omitted_inputs = omissions[name]
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(previous_mode)


def paired_full_comparison(before, after, *, before_directory: Path, after_directory: Path):
    """Compare matching streamed cases without retaining the rows in memory."""
    if before.keys() != after.keys():
        raise ValueError("paired evaluation requires the same complete sources")
    result = {}
    for name, previous in before.items():
        following = after[name]
        if not previous["complete"] or not following["complete"] or previous["case_sha256"] != following["case_sha256"]:
            raise ValueError("paired full evaluation requires matching complete cases")
        totals = MetricTotals()
        with (before_directory / previous["rows_file"]).open() as old_file, (after_directory / following["rows_file"]).open() as new_file:
            for old_line, new_line in zip_longest(old_file, new_file):
                if old_line is None or new_line is None:
                    raise ValueError("paired full evaluation has different row counts")
                old, new = json.loads(old_line), json.loads(new_line)
                identity = ("key", "group", "record_key", "form", "wording")
                if any(old[key] != new[key] for key in identity) or old["metrics"].keys() != new["metrics"].keys():
                    raise ValueError("paired full evaluation has different cases or metrics")
                if (old.get("response") or {}).get("record_indices") != (new.get("response") or {}).get("record_indices"):
                    raise ValueError("paired full evaluation changed comparison partners")
                totals.add({key: new["metrics"][key] - value for key, value in old["metrics"].items()})
        if totals.counts.get("loss") != previous["examples"] or previous["examples"] != following["examples"]:
            raise ValueError("paired full evaluation rows differ from their manifests")
        result[name] = {"before": previous["summary"], "after": following["summary"],
                        "paired_change": totals.summary(), "examples": previous["examples"], "groups": previous["groups"]}
    return result
