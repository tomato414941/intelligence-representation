from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import defaultdict

import torch

from intrep.problems.shared_prediction.sources import (
    ClassificationSource,
    NativeSource,
    ShogiSource,
    TextSource,
)


def make_panel(sources, examples: int, seed: int = 9047) -> dict:
    """Select fixed validation locations without consuming any training stream."""
    if examples < 1:
        raise ValueError("evaluation requires a positive example count")
    generator = random.Random(seed)
    panel = {}
    for name, source in sources.items():
        cases = []
        if isinstance(source, (TextSource, ShogiSource)):
            stream = source.stream
            offsets, records = [], 0
            with stream.path.open("rb") as handle:
                handle.seek(stream.start)
                while handle.tell() < stream.end:
                    offset = handle.tell()
                    if not handle.readline().strip():
                        continue
                    records += 1
                    if len(offsets) < examples:
                        offsets.append(offset)
                    else:
                        replacement = generator.randrange(records)
                        if replacement < examples:
                            offsets[replacement] = offset
            if not offsets:
                raise ValueError(f"source {name!r} has no evaluation records")
            cases = [{"offset": offset, "key": f"byte:{offset}"} for offset in sorted(offsets)]
        elif isinstance(source, ClassificationSource):
            positions = generator.sample(range(source.sampler.count), min(examples, source.sampler.count))
            cases = [{"cursor": cursor, "key": f"image:{int(source.sampler.order[cursor])}"} for cursor in positions]
        elif isinstance(source, NativeSource):
            # Native counts refer to independent worlds, with all their transitions.
            original = copy.deepcopy(source.state_dict())
            try:
                for episode in generator.sample(range(len(source.entries)), min(examples, len(source.entries))):
                    source._load(episode)
                    for transition in range(len(source.episode.actions)):
                        cases.append({"episode": episode, "transition": transition,
                                      "key": f"{source.episode.id}:{transition}", "group": source.episode.world_id})
            finally:
                source.load_state_dict(original)
        elif callable(getattr(source, "evaluation_cases", None)):
            cases = source.evaluation_cases(examples, generator)
        else:
            raise TypeError(f"source {name!r} must implement an evaluation panel for its own format")
        panel[name] = cases
    return panel


def set_case(source, case):
    if isinstance(source, (TextSource, ShogiSource)):
        source.stream.load_state_dict({"offset": case["offset"], "epochs": 0, "records": 0})
        if isinstance(source, TextSource):
            source.pending = []
            source.tokens = 0
            if hasattr(source, "pending_mask"):
                source.pending_mask = []
                source.position = 0
    elif isinstance(source, ClassificationSource):
        source.sampler.cursor = case["cursor"]
    elif isinstance(source, NativeSource):
        if source.active != case["episode"] or source.episode is None:
            source._load(case["episode"])
        source.transition = case["transition"]
    else:
        source.set_evaluation_case(case)


def summarize(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in row["metrics"].items():
            if not math.isfinite(value):
                raise ValueError("evaluation produced a nonfinite metric")
            values[key].append(value)
    return {key: {"mean": sum(samples) / len(samples), "count": len(samples)} for key, samples in values.items()}


@torch.no_grad()
def evaluate_panel(model, sources, panel, *, omit_native=(), max_native_worlds=None, generate_answers=True):
    states = {name: copy.deepcopy(source.state_dict()) for name, source in sources.items()}
    native = {name: getattr(source, "reader", source) for name, source in sources.items()
              if isinstance(getattr(source, "reader", source), NativeSource)}
    omissions = {name: source.omitted_inputs.copy() for name, source in native.items()}
    previous_mode = model.training
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []
    model.eval()
    result = {}
    try:
        for name, cases in panel.items():
            source = sources[name]
            rows, worlds = [], set()
            if name in native:
                native[name].omitted_inputs = set(omit_native)
            for case in cases:
                group = case.get("group", case["key"])
                if name in native and max_native_worlds is not None:
                    if group not in worlds and len(worlds) >= max_native_worlds:
                        continue
                    worlds.add(group)
                set_case(source, {**case, "generate": generate_answers and case.get("generate", True)})
                loss = source.loss()
                group = getattr(source, "last_group", None) or group
                metrics = {"loss": float(loss), **{key: float(value) for key, value in getattr(source, "last_metrics", {}).items()}}
                row = {"key": case["key"], "group": group, "metrics": metrics}
                if "form" in case:
                    row.update(form=case["form"], wording=case["wording"], record_key=case["record_key"])
                    row["response"] = copy.deepcopy(getattr(source, "last_response", None))
                rows.append(row)
            result[name] = {"summary": summarize(rows), "rows": rows,
                            "groups": len({row["group"] for row in rows})}
            if any("form" in row for row in rows):
                forms = {(row["form"], row["wording"]) for row in rows}
                result[name]["forms"] = {f"{form}/{wording}": summarize([row for row in rows if (row["form"], row["wording"]) == (form, wording)])
                                          for form, wording in sorted(forms)}
        return result
    finally:
        for name, state in states.items():
            sources[name].load_state_dict(state)
            if name in omissions:
                native[name].omitted_inputs = omissions[name]
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(previous_mode)


def paired_comparison(before, after):
    if set(before) != set(after):
        raise ValueError("paired evaluation requires the same sources")
    result = {}
    for name in before:
        original = {row["key"]: row for row in before[name]["rows"]}
        following = {row["key"]: row for row in after[name]["rows"]}
        if original.keys() != following.keys():
            raise ValueError("paired evaluation requires the same examples")
        deltas = []
        for key, old in original.items():
            new = following[key]
            if old["metrics"].keys() != new["metrics"].keys():
                raise ValueError("paired evaluation metric identities differ")
            deltas.append({"metrics": {metric: new["metrics"][metric] - value for metric, value in old["metrics"].items()}})
        result[name] = {"before": before[name]["summary"], "after": after[name]["summary"],
                        "paired_change": summarize(deltas), "examples": len(deltas), "groups": before[name]["groups"]}
    return result


def question_omission_panel(sources, panel, examples=16):
    result = {}
    for name, cases in panel.items():
        source = sources[name]
        if not hasattr(source, "reader") or source.config["kind"] in ("text", "conversations", "boolq"):
            continue
        candidates = [case for case in cases if case["form"] != "original" and case["wording"] == 0]
        keys = sorted({case["record_key"] for case in candidates}, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
        selected = set(keys[:examples])
        result[name] = [{**case, "omit_observations": True} for case in candidates if case["record_key"] in selected]
    return result
