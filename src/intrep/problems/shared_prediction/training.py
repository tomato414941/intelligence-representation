from __future__ import annotations

import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.evaluation import (
    evaluate_panel,
    make_panel,
    paired_comparison,
    question_omission_panel,
)
from intrep.problems.shared_prediction.recipe import (
    evaluation_recipe,
    validate_extension,
    validate_recipe,
)
from intrep.problems.shared_prediction.sources import (
    build_sources,
    configure_heads,
    source_configs,
)
from intrep.representation.cores.lfm import create_lfm, load_lfm

SCHEMA = "intrep.shared_prediction_checkpoint.v1"


def save_checkpoint(path: Path, model, trainer, sources, recipe, provenance, tokenizer, extensions=()):
    payload = {"schema_version": SCHEMA, "lfm_config": model.core.body.config.to_dict(),
               "attention_implementation": model.core.body.config._attn_implementation,
               "modules": model.module_state_dict(), "trainer": trainer.state_dict(),
               "sources": {name: source.state_dict() for name, source in sources.items()},
               "recipe": recipe, "provenance": provenance, "extensions": list(extensions),
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else [],
               "dtype": str(next(model.parameters()).dtype).removeprefix("torch.")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path.parent / "tokenizer")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, *, device="cpu", extensions=(), extend=False):
    from transformers import AutoTokenizer

    payload = torch.load(path, map_location="cpu", weights_only=True)
    previous_extensions = payload.get("extensions", [])
    compatible_extensions = (list(extensions)[:len(previous_extensions)] == previous_extensions
                             if extend else list(extensions) == previous_extensions)
    if payload.get("schema_version") != SCHEMA or not compatible_extensions:
        raise ValueError("checkpoint schema or explicitly loaded source extensions differ")
    model = create_lfm(payload["lfm_config"], device=device,
                       attention_implementation=payload["attention_implementation"]).to(dtype=getattr(torch, payload["dtype"]))
    configure_heads(model, payload["recipe"])
    model.load_module_state_dict(payload["modules"])
    del payload["modules"]
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer", local_files_only=True)
    return model, tokenizer, payload


def parameter_digests(model):
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters()}


class GradientAudit:
    def __init__(self, model):
        self.source = None
        self.records = {}
        self.handles = []
        for index, layer in enumerate(model.core.body.layers):
            operator = layer.self_attn.q_proj if layer.is_attention_layer else layer.conv.in_proj
            for kind, parameter in (("operator", operator.weight), ("feed_forward", layer.feed_forward.w2.weight)):
                key = f"layer_{index}.{kind}"
                self.handles.append(parameter.register_hook(lambda gradient, key=key: self.record(key, gradient)))

    def record(self, key, gradient):
        valid = bool(torch.isfinite(gradient).all() and torch.count_nonzero(gradient))
        self.records.setdefault(self.source, {})[key] = self.records.get(self.source, {}).get(key, False) or valid

    def close(self):
        for handle in self.handles:
            handle.remove()


class SourceGradientProbe:
    """Observe weighted gradients on two body projections without extra updates."""

    def __init__(self, model):
        self.enabled = False
        self.source = None
        self.values = {}
        self.handles = []
        self.parameters = []
        layers = model.core.body.layers
        for index in sorted({0, len(layers) - 1}):
            layer = layers[index]
            operator = layer.self_attn.q_proj if layer.is_attention_layer else layer.conv.in_proj
            key = f"layer_{index}.{'attention_q' if layer.is_attention_layer else 'conv_in'}"
            self.parameters.append(key)
            self.handles.append(operator.weight.register_hook(lambda gradient, key=key: self.record(key, gradient)))

    def begin(self, enabled):
        self.enabled = enabled
        self.values = {}

    def record(self, key, gradient):
        if self.enabled:
            self.values.setdefault(self.source, {})[key] = gradient.detach().float().clone()

    def summary(self, weights):
        if not self.enabled:
            return None
        norms = {name: sum(value.square().sum() for value in values.values()).sqrt()
                 for name, values in self.values.items()}
        rows = {name: {"weighted_norm": float(norm),
                       "unweighted_norm": float(norm) * sum(weights.values()) / weights[name]}
                for name, norm in norms.items()}
        reference = self.values.get("conversations")
        if reference:
            for name, values in self.values.items():
                dot = sum((values[key] * other).sum() for key, other in reference.items())
                denominator = norms[name] * norms["conversations"]
                rows[name]["cosine_to_conversations"] = float(dot / denominator) if float(denominator) > 0 else None
        self.values = {}
        return {"parameters": self.parameters, "sources": rows,
                "scope": "Selected first/last body projections, before global gradient clipping; not whole-model norms."}

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, *, max_tokens=32):
    device = next(model.parameters()).device
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                        add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)
    result = []
    previous_mode = model.training
    model.eval()
    try:
        for _ in range(max_tokens):
            hidden = model(model.encode("text", ids))
            token = int(model.decode("text", hidden[:, -1:])[0, 0].argmax())
            if token == tokenizer.eos_token_id:
                break
            result.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        return tokenizer.decode(result, skip_special_tokens=True)
    finally:
        model.train(previous_mode)


def train(*, base: Path | None, recipe: dict, root: Path, output: Path, steps: int,
          device="cpu", optimizer="sgd", learning_rate=0.0001, max_grad_norm=1.0,
          resume: Path | None = None, extend=False, audit_gradients=False, extensions=(), checkpoint_interval=100,
          evaluation_examples=1, evaluation_interval=0, native_controls=False, prompts=None, training_seconds=None,
          generation_interval=0, holdout_prompts=None, gradient_probe_interval=0):
    from transformers import AutoTokenizer

    if (steps < 1 or checkpoint_interval < 1 or evaluation_examples < 1 or evaluation_interval < 0 or generation_interval < 0
            or gradient_probe_interval < 0
            or (base is None) == (resume is None) or (extend and resume is None)):
        raise ValueError("choose a local base or a resume checkpoint and a positive total step budget")
    if training_seconds is not None and (not 0 < training_seconds < float("inf")):
        raise ValueError("the training-time budget must be finite and positive")
    validate_recipe(recipe, root)
    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists() and (resume is None or checkpoint_path.resolve() != resume.resolve()):
        raise FileExistsError("output already contains a different training checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    payload = None
    if resume is not None:
        model, tokenizer, payload = load_checkpoint(resume, device=device, extensions=extensions, extend=extend)
        if extend:
            validate_extension(payload["recipe"], recipe)
        elif payload["recipe"] != recipe:
            raise ValueError("exact resume requires the original complete data recipe")
        old_weights = payload["trainer"]["weights"]
        trainer = JointTrainer(model, old_weights, learning_rate=learning_rate, optimizer=optimizer,
                               max_grad_norm=max_grad_norm)
        trainer.load_state_dict(payload["trainer"])
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            if not torch.cuda.is_available() or device == "cpu":
                raise ValueError("exact CUDA continuation requires its original device family")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    else:
        torch.manual_seed(recipe.get("seed", 47))
        model = load_lfm(str(base), device=device)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        configure_heads(model, recipe)
        weights = {row["name"]: row.get("weight", 1.0) for row in source_configs(recipe)}
        trainer = JointTrainer(model, weights, learning_rate=learning_rate, optimizer=optimizer,
                               max_grad_norm=max_grad_norm)
    sources = build_sources(model, tokenizer, recipe, root)
    evaluation_sources = build_sources(model, tokenizer, evaluation_recipe(recipe), root)
    print(json.dumps({"stage": "checking_source_provenance", "sources": list(sources)}), flush=True)
    provenance = {name: source.provenance() for name, source in sources.items()}
    evaluation_provenance = {name: source.provenance() for name, source in evaluation_sources.items()}
    if payload is not None:
        for name, previous in payload["provenance"].items():
            if provenance.get(name) != previous:
                raise ValueError("previous training populations must remain unchanged on continuation")
            sources[name].load_state_dict(payload["sources"][name])
        for row in source_configs(recipe):
            if row["name"] not in trainer.weights:
                trainer.add_source(row["name"], row.get("weight", 1.0))
    if steps <= trainer.steps:
        raise ValueError("the requested total step budget must advance the checkpoint")
    trainer.synchronize_parameters()
    (output / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({"training": provenance, "evaluation": evaluation_provenance},
                                                      ensure_ascii=False, indent=2) + "\n")
    before = parameter_digests(model)
    panel = make_panel(evaluation_sources, evaluation_examples)
    (output / "evaluation-panel.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n")
    prompts = prompts if prompts is not None else [
        {"prompt": "Explain why ice melts in one sentence."},
        {"prompt": "氷が溶ける理由を日本語で一文で説明してください。"},
    ]

    def measure_generations(*, include_holdout=False):
        rows = []
        for split, cases in (("development", prompts), ("holdout", holdout_prompts or [])):
            if split == "holdout" and not include_holdout:
                continue
            for case in cases:
                answer = generate_text(model, tokenizer, case["prompt"], max_tokens=case.get("max_tokens", 48))
                row = {**case, "split": split, "answer": answer}
                if "expected" in case:
                    row["exact_match"] = answer.strip() == case["expected"]
                rows.append(row)
        directory = output / "generations"
        directory.mkdir(exist_ok=True)
        (directory / f"step-{trainer.steps:06d}.json").write_text(json.dumps(
            {"step": trainer.steps, "generations": rows}, ensure_ascii=False, indent=2) + "\n")
        scored = [row for row in rows if row["split"] == "development" and "exact_match" in row]
        print(json.dumps({"stage": "generation", "step": trainer.steps,
                          "development_correct": sum(row["exact_match"] for row in scored),
                          "development_count": len(scored)}), flush=True)
        return rows

    def measure(*, controls=False, generations=False):
        measured = evaluate_panel(model, evaluation_sources, panel, generate_answers=generations)
        report = {"step": trainer.steps, "sources": measured}
        if controls and native_controls:
            from intrep.problems.shared_prediction.sources import NativeSource
            native_panel = {name: cases for name, cases in panel.items()
                            if isinstance(getattr(evaluation_sources[name], "reader", evaluation_sources[name]), NativeSource)}
            report["native_input_controls"] = {}
            for omission in (None, "image", "audio", "text"):
                report["native_input_controls"][omission or "complete"] = evaluate_panel(
                    model, evaluation_sources, native_panel, omit_native=() if omission is None else (omission,), max_native_worlds=16,
                    generate_answers=False,
                )
            omitted = question_omission_panel(evaluation_sources, panel)
            if omitted:
                report["question_without_observations"] = evaluate_panel(model, evaluation_sources, omitted, generate_answers=generations)
        if generations:
            report["generations"] = measure_generations(include_holdout=True)
        directory = output / "evaluation"
        directory.mkdir(exist_ok=True)
        (directory / f"step-{trainer.steps:06d}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": "evaluation", "step": trainer.steps,
                          "losses": {name: value["summary"]["loss"] for name, value in measured.items()}}), flush=True)
        return report

    before_evaluation = measure(controls=True, generations=True)
    audit = GradientAudit(model) if audit_gradients else None
    probe = SourceGradientProbe(model) if gradient_probe_interval else None

    def source_loss(name, source):
        if audit:
            audit.source = name
        if probe:
            probe.source = name
        return source.loss()

    records = []
    elapsed_training = 0.0
    try:
        while trainer.steps < steps:
            if training_seconds is not None and elapsed_training >= training_seconds:
                break
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            if probe:
                next_step = trainer.steps + 1
                probe.begin(next_step <= 2 or next_step % gradient_probe_interval in (0, 1))
            callbacks = {name: (lambda name=name, source=source: source_loss(name, source)) for name, source in sources.items()}
            metrics = trainer.step(callbacks)
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            record = {"step": trainer.steps, "seconds": time.perf_counter() - start, **metrics}
            if probe and probe.enabled:
                record["gradient_probe"] = probe.summary(trainer.weights)
            elapsed_training += record["seconds"]
            if any(hasattr(source, "last_update_info") for source in sources.values()):
                record["source_details"] = {name: source.last_update_info for name, source in sources.items()
                                             if hasattr(source, "last_update_info")}
            records.append(record)
            with (output / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if trainer.steps % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, model, trainer, sources, recipe, provenance, tokenizer, extensions)
            if evaluation_interval and trainer.steps % evaluation_interval == 0 and trainer.steps < steps:
                measure()
            if generation_interval and trainer.steps % generation_interval == 0 and trainer.steps < steps:
                measure_generations()
    finally:
        if audit:
            audit.close()
        if probe:
            probe.close()
    after = parameter_digests(model)
    changed = [name for name in before if before[name] != after[name]]
    core_names = [name for name in before if name.startswith("core.")]
    save_checkpoint(checkpoint_path, model, trainer, sources, recipe, provenance, tokenizer, extensions)
    after_evaluation = measure(controls=True, generations=True)
    result = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
              "initial_parameters_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
              "final_parameters_sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
              "requested_steps": steps, "completed_steps": trainer.steps,
              "training_seconds_budget": training_seconds, "training_seconds": elapsed_training,
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "core_parameter_tensors": len(core_names),
              "changed_core_parameter_tensors": sum(name in changed for name in core_names),
              "changed_parameter_names": changed, "joint_updates": records,
              "source_progress": {name: source.progress() for name, source in sources.items()
                                  if hasattr(source, "progress")},
              "gradient_audit": audit.records if audit else None,
              "evaluation_before": {name: row["summary"]["loss"]["mean"] for name, row in before_evaluation["sources"].items()},
              "evaluation_after": {name: row["summary"]["loss"]["mean"] for name, row in after_evaluation["sources"].items()},
              "paired_evaluation": paired_comparison(before_evaluation["sources"], after_evaluation["sources"]),
              "text_before": {row["prompt"]: row["answer"] for row in before_evaluation["generations"]},
              "text_after": {row["prompt"]: row["answer"] for row in after_evaluation["generations"]},
              "max_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "torch_version": str(torch.__version__), "device": device,
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
              "input_heads": list(model.input_heads), "output_heads": list(model.output_heads),
              "limitations": "Fixed development panel; descriptive paired changes are not full-population evaluation or causal cross-task transfer evidence."}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "saved", "checkpoint": str(checkpoint_path),
                      "changed_core_tensors": result["changed_core_parameter_tensors"],
                      "core_tensors": len(core_names)}), flush=True)
    return checkpoint_path
