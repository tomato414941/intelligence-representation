from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
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

    def loss(self, name, source):
        self.source = name
        return source.loss()

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def evaluate(model, sources):
    states = {name: copy.deepcopy(source.state_dict()) for name, source in sources.items()}
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else []
    previous_mode = model.training
    model.eval()
    try:
        return {name: float(source.loss()) for name, source in sources.items()}
    finally:
        for name, state in states.items():
            sources[name].load_state_dict(state)
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(previous_mode)


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
          resume: Path | None = None, extend=False, audit_gradients=False, extensions=(), checkpoint_interval=100):
    from transformers import AutoTokenizer

    if steps < 1 or checkpoint_interval < 1 or (base is None) == (resume is None) or (extend and resume is None):
        raise ValueError("choose a local base or a resume checkpoint and a positive total step budget")
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
    before_evaluation = evaluate(model, evaluation_sources)
    prompts = ["Explain why ice melts in one sentence.", "氷が溶ける理由を日本語で一文で説明してください。"]
    before_text = {prompt: generate_text(model, tokenizer, prompt) for prompt in prompts}
    audit = GradientAudit(model) if audit_gradients else None
    records = []
    try:
        while trainer.steps < steps:
            start = time.perf_counter()
            callbacks = {name: (lambda name=name, source=source: audit.loss(name, source)) if audit else source.loss
                         for name, source in sources.items()}
            metrics = trainer.step(callbacks)
            record = {"step": trainer.steps, "seconds": time.perf_counter() - start, **metrics}
            records.append(record)
            with (output / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if trainer.steps % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, model, trainer, sources, recipe, provenance, tokenizer, extensions)
    finally:
        if audit:
            audit.close()
    after = parameter_digests(model)
    changed = [name for name in before if before[name] != after[name]]
    core_names = [name for name in before if name.startswith("core.")]
    save_checkpoint(checkpoint_path, model, trainer, sources, recipe, provenance, tokenizer, extensions)
    result = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "core_parameter_tensors": len(core_names),
              "changed_core_parameter_tensors": sum(name in changed for name in core_names),
              "changed_parameter_names": changed, "joint_updates": records,
              "source_progress": {name: source.progress() for name, source in sources.items()
                                  if hasattr(source, "progress")},
              "gradient_audit": audit.records if audit else None,
              "evaluation_before": before_evaluation, "evaluation_after": evaluate(model, evaluation_sources),
              "text_before": before_text, "text_after": {prompt: generate_text(model, tokenizer, prompt) for prompt in prompts},
              "max_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "torch_version": str(torch.__version__), "device": device,
              "input_heads": list(model.input_heads), "output_heads": list(model.output_heads),
              "limitations": "Execution and development-loss measurements; not full-population epoch completion or a capability claim."}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "saved", "checkpoint": str(checkpoint_path),
                      "changed_core_tensors": result["changed_core_parameter_tensors"],
                      "core_tensors": len(core_names)}), flush=True)
    return checkpoint_path
