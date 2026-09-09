from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from intrep.core.training_utils import build_adamw, resolve_training_device
from intrep.problems.cellular_rule_inference.episodes import (
    deserialize_rules,
    sample_episodes,
    sample_rules,
    serialize_rules,
)
from intrep.representation.assemblies.cellular_rule_inference import (
    CellularRuleInferenceModel,
    CellularRuleInferenceModelConfig,
)

SCHEMA = "intrep.cellular_rule_inference_checkpoint.v1"


@dataclass(frozen=True)
class RuleInferenceTrainingConfig:
    model: CellularRuleInferenceModelConfig = field(default_factory=CellularRuleInferenceModelConfig)
    train_rule_count: int = 256
    rule_seed: int = 1701
    data_seed: int = 3101
    model_seed: int = 31
    max_steps: int = 2000
    batch_size: int = 16
    learning_rate: float = 0.0003
    warmup_steps: int = 100


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[CellularRuleInferenceModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported cellular rule inference checkpoint")
    model = CellularRuleInferenceModel(CellularRuleInferenceModelConfig(**payload["config"]["model"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(resolve_training_device(device)), payload


def train(config: RuleInferenceTrainingConfig, run_dir: Path, *, device: str = "auto", resume: bool = False) -> Path:
    if config.max_steps < 1 or config.batch_size < 1 or config.learning_rate <= 0 or config.warmup_steps < 0:
        raise ValueError("invalid training budget or learning rate")
    if config.model.max_context != 8:
        raise ValueError("this experiment uses the fixed context sweep 0, 1, 4, 8")
    resolved = resolve_training_device(device)
    torch.manual_seed(config.model_seed)
    rng = np.random.default_rng(config.data_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "checkpoint.pt"
    serialized_config = asdict(config)
    start_step = 0
    payload = None
    if resume:
        model, payload = load_checkpoint(path, device)
        old = {k: v for k, v in payload["config"].items() if k != "max_steps"}
        new = {k: v for k, v in serialized_config.items() if k != "max_steps"}
        if old != new:
            raise ValueError("resume may change only max_steps, not the experiment configuration")
        rules = deserialize_rules(payload["train_rules"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        start_step = payload["step"]
    else:
        if path.exists():
            raise FileExistsError("checkpoint exists; use resume or a new run directory")
        rules = sample_rules(config.train_rule_count, config.rule_seed)
        model = CellularRuleInferenceModel(config.model).to(resolved)
    optimizer = build_adamw(model, learning_rate=config.learning_rate, weight_decay=0.01)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
    (run_dir / "config.json").write_text(json.dumps(serialized_config, indent=2) + "\n")
    model.train()
    started = time.perf_counter()
    losses = []
    for step in range(start_step + 1, config.max_steps + 1):
        count = int(rng.choice([0, 1, 1, 4, 4, 8, 8, 8]))
        batch_rules = [rules[int(index)] for index in rng.integers(len(rules), size=config.batch_size)]
        episodes = sample_episodes(batch_rules, rng, height=config.model.height, width=config.model.width, context_count=count)
        support = torch.as_tensor(episodes.support, dtype=torch.float32, device=resolved)
        query = torch.as_tensor(episodes.query, dtype=torch.float32, device=resolved)
        targets = torch.as_tensor(episodes.targets, dtype=torch.long, device=resolved)
        learning_rate = config.learning_rate * min(1.0, step / max(1, config.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(resolved.type, dtype=torch.bfloat16, enabled=resolved.type == "cuda"):
            logits = model(support, query)
            loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == start_step + 1 or step % 100 == 0 or step == config.max_steps:
            row = {"step": step, "mean_loss": float(np.mean(losses)), "last_context_count": count,
                   "elapsed_seconds": time.perf_counter() - started, "learning_rate": learning_rate}
            with (run_dir / "training.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            losses.clear()
        if step % 500 == 0 or step == config.max_steps:
            checkpoint = {"schema_version": SCHEMA, "config": serialized_config, "step": step,
                          "train_rules": serialize_rules(rules), "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "numpy_rng_state": rng.bit_generator.state,
                          "torch_rng_state": torch.get_rng_state(), "device": str(resolved),
                          "autocast_dtype": "bfloat16" if resolved.type == "cuda" else "float32"}
            temporary = path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
    return path
