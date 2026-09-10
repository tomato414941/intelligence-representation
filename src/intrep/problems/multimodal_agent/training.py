from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from intrep.core.training_utils import build_adamw, resolve_training_device
from intrep.experience.multimodal.records import MultimodalEpisode, selected_episodes
from intrep.learning.replay_buffer import ReplayBuffer
from intrep.representation.assemblies.multimodal_agent import (
    MultimodalAgentConfig,
    MultimodalAgentModel,
)

SCHEMA = "intrep.multimodal_agent_checkpoint.v1"


@dataclass(frozen=True)
class MultimodalTrainingConfig:
    model: MultimodalAgentConfig = field(default_factory=MultimodalAgentConfig)
    steps: int = 3000
    batch_size: int = 16
    replay_capacity: int = 4096
    learning_rate: float = 0.0003
    warmup_steps: int = 100
    seed: int = 41
    discount: float = 0.95
    target_update_rate: float = 0.01
    teacher_weight: float = 1.0
    value_weight: float = 0.5
    text_weight: float = 0.5
    image_weight: float = 4.0
    audio_weight: float = 2.0
    feedback_weight: float = 0.25

    def __post_init__(self) -> None:
        if (min(self.steps, self.batch_size, self.replay_capacity) < 1 or self.learning_rate <= 0
                or not 0 <= self.discount <= 1 or not 0 < self.target_update_rate <= 1 or self.warmup_steps < 0):
            raise ValueError("invalid training configuration")
        if any(getattr(self, name) < 0 for name in (
            "teacher_weight", "value_weight", "text_weight", "image_weight", "audio_weight", "feedback_weight",
        )):
            raise ValueError("loss weights must be nonnegative")


@torch.no_grad()
def target_values(model: MultimodalAgentModel, episodes: Sequence[MultimodalEpisode]) -> list[list[torch.Tensor]]:
    memory = model.new_memory(len(episodes))
    result: list[list[torch.Tensor]] = [[] for _ in episodes]
    for step in range(max(len(episode.observations) for episode in episodes)):
        active = [index for index, episode in enumerate(episodes) if step < len(episode.observations)]
        indices = torch.tensor(active, device=memory.device)
        updated = model.observe([episodes[index].observations[step] for index in active], memory[indices], step=step)
        memory = memory.index_copy(0, indices, updated)
        values = model.policy(updated)
        for index, value in zip(active, values):
            result[index].append(value.max())
    return result


def episode_loss(
    model: MultimodalAgentModel, episodes: Sequence[MultimodalEpisode], config: MultimodalTrainingConfig,
    *, target_model: MultimodalAgentModel | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not episodes:
        raise ValueError("training batch must contain episodes")
    for episode in episodes:
        episode.validate()
    targets = target_values(target_model, episodes) if target_model is not None else None
    memory = model.new_memory(len(episodes))
    device = memory.device
    components: dict[str, list[torch.Tensor]] = {key: [] for key in ("teacher", "value", "text", "image", "audio", "feedback")}
    for step in range(max(len(episode.actions) for episode in episodes)):
        active = [index for index, episode in enumerate(episodes) if step < len(episode.actions)]
        indices = torch.tensor(active, device=device)
        observations = [episodes[index].observations[step] for index in active]
        next_observations = [episodes[index].observations[step + 1] for index in active]
        updated = model.observe(observations, memory[indices], step=step)
        memory = memory.index_copy(0, indices, updated)
        values = model.policy(updated)
        actions = torch.tensor([episodes[index].actions[step] for index in active], device=device)
        supervised = [(local, episodes[index].teacher_actions[step]) for local, index in enumerate(active)
                      if episodes[index].teacher_actions and episodes[index].teacher_actions[step] is not None]
        if supervised:
            rows, labels = zip(*supervised)
            components["teacher"].append(F.cross_entropy(values[list(rows)], torch.tensor(labels, device=device)))
        if targets is not None:
            expected = torch.stack([
                next_observation.feedback[0].to(device)
                + config.discount * (1 - next_observation.feedback[1].to(device)) * targets[index][step + 1]
                for index, next_observation in zip(active, next_observations)
            ])
            components["value"].append(F.smooth_l1_loss(values.gather(1, actions[:, None]).squeeze(1).float(), expected.float()))
        answers = [(local, episodes[index].answers[step]) for local, index in enumerate(active)
                   if episodes[index].answers and episodes[index].answers[step] is not None]
        if answers:
            rows, texts = zip(*answers)
            components["text"].append(model.text_loss(updated[list(rows)], texts))
        # Waveform time coordinates depend on sample rate; group those requests explicitly.
        rates = sorted({observation.sample_rate for observation in next_observations})
        for rate in rates:
            rows = [index for index, observation in enumerate(next_observations) if observation.sample_rate == rate]
            following = [next_observations[index] for index in rows]
            audio_samples = max((len(observation.audio) if observation.audio is not None else 0) for observation in following)
            shapes = [tuple(observation.image.shape[:2]) if observation.image is not None else None for observation in following]
            prediction = model.predict_outcome(updated[rows], actions[rows], image_shapes=shapes,
                                               audio_samples=audio_samples, sample_rate=rate)
            feedback = torch.stack([observation.feedback.to(device) for observation in following])
            components["feedback"].append(F.mse_loss(prediction.feedback[:, 0], feedback[:, 0])
                                            + F.binary_cross_entropy_with_logits(prediction.feedback[:, 1:], feedback[:, 1:]))
            for local, row in enumerate(rows):
                future, previous = next_observations[row], observations[row]
                if future.image is not None:
                    expected_image = future.image.to(device)
                    weight = torch.ones_like(expected_image[..., :1])
                    if previous.image is not None and previous.image.shape == future.image.shape:
                        changed = (future.image - previous.image).abs().amax(-1, keepdim=True) > 0.01
                        weight = weight + 4 * changed.to(device)
                    error = (prediction.images[local] - expected_image).square()
                    components["image"].append((error * weight).sum() / (weight.sum() * 3))
                if future.audio is not None:
                    components["audio"].append(F.mse_loss(prediction.audio[local, :len(future.audio)], future.audio.to(device)))
    zero = memory.sum() * 0
    averages = {name: torch.stack(losses).mean() if losses else zero for name, losses in components.items()}
    total = sum(getattr(config, f"{name}_weight") * value for name, value in averages.items())
    return total, {name: float(value.detach()) for name, value in averages.items()}


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[MultimodalAgentModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported multimodal agent checkpoint")
    model = MultimodalAgentModel(MultimodalAgentConfig(**payload["config"]["model"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(resolve_training_device(device)), payload


def train(
    selections: Sequence[Path], output: Path, config: MultimodalTrainingConfig, *, device: str = "auto",
    resume: bool = False, initialize: Path | None = None,
) -> Path:
    if resume and initialize is not None:
        raise ValueError("choose either exact resume or initialization from learned weights")
    resolved = resolve_training_device(device)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)
    source_records, episodes = [], []
    identities: set[str] = set()
    world_partitions: dict[str, str] = {}
    for selection in selections:
        for entry in json.loads(selection.read_text())["episodes"]:
            if world_partitions.get(entry["world_id"], entry["split"]) != entry["split"]:
                raise ValueError("combined selections leak a world across train and evaluation")
            world_partitions[entry["world_id"]] = entry["split"]
        selected = selected_episodes(selection, "train")
        for episode in selected:
            if episode.id in identities:
                raise ValueError("training selections contain duplicate episodes")
            identities.add(episode.id)
        episodes.extend(selected)
        source_records.append({"selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
                               "episode_ids": [episode.id for episode in selected],
                               "world_ids": [episode.world_id for episode in selected]})
    if len(episodes) < config.batch_size:
        raise ValueError("training selection is smaller than a replay batch")
    if config.replay_capacity < len(episodes):
        raise ValueError("replay capacity would evict explicitly selected source episodes")
    buffer = ReplayBuffer[MultimodalEpisode](capacity=config.replay_capacity)
    buffer.extend(episodes)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "checkpoint.pt"
    start_step, payload, initialization = 0, None, None
    training_history = list(source_records)
    if resume:
        model, payload = load_checkpoint(path, device=device)
        old = {key: value for key, value in payload["config"].items() if key != "steps"}
        new = {key: value for key, value in asdict(config).items() if key != "steps"}
        if old != new or source_records != payload["sources"]:
            raise ValueError("exact resume may change only the number of steps")
        start_step = payload["step"]
        if config.steps < start_step:
            raise ValueError("resume step budget cannot precede the saved checkpoint")
        initialization = payload.get("initialization")
        training_history = payload.get("training_history", payload["sources"])
        generator.set_state(payload["replay_rng"])
        torch.set_rng_state(payload["torch_rng"])
    else:
        if path.exists():
            raise FileExistsError("checkpoint exists; use resume or a new output directory")
        if initialize is None:
            model = MultimodalAgentModel(config.model).to(resolved)
        else:
            model, initialized = load_checkpoint(initialize, device=device)
            if model.config != config.model:
                raise ValueError("initialization model configuration differs")
            initialization = {"checkpoint_sha256": hashlib.sha256(initialize.read_bytes()).hexdigest(),
                              "step": initialized["step"]}
            for source in initialized.get("training_history", initialized["sources"]):
                if source not in training_history:
                    training_history.append(source)
            inherited_worlds = {world for source in training_history for world in source["world_ids"]}
            if any(split != "train" and world in inherited_worlds for world, split in world_partitions.items()):
                raise ValueError("initialization already learned a declared evaluation world")
    target_model = copy.deepcopy(model).eval()
    target_model.requires_grad_(False)
    optimizer = build_adamw(model, learning_rate=config.learning_rate, weight_decay=0.01)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        target_model.load_state_dict(payload["target_model"], strict=True)
        if resolved.type == "cuda" and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "sources.json").write_text(json.dumps(source_records, indent=2) + "\n")
    model.train()
    started = time.perf_counter()
    running: list[dict[str, float]] = []
    for step in range(start_step + 1, config.steps + 1):
        batch = buffer.sample(config.batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        lr = config.learning_rate * min(1.0, step / max(1, config.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(resolved.type, dtype=torch.bfloat16, enabled=resolved.type == "cuda"):
            loss, metrics = episode_loss(model, batch, config, target_model=target_model)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite multimodal training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite multimodal training gradient")
        optimizer.step()
        with torch.no_grad():
            for target, current in zip(target_model.parameters(), model.parameters()):
                target.lerp_(current, config.target_update_rate)
        running.append({"loss": float(loss.detach()), **metrics})
        if step == start_step + 1 or step % 50 == 0 or step == config.steps:
            row = {"step": step, "elapsed_seconds": time.perf_counter() - started, "learning_rate": lr,
                   **{key: sum(item[key] for item in running) / len(running) for key in running[0]}}
            with (output / "training.jsonl").open("a") as target:
                target.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            running.clear()
        if step % 250 == 0 or step == config.steps:
            state = {
                "schema_version": SCHEMA, "config": asdict(config), "step": step, "sources": source_records,
                "training_history": training_history,
                "initialization": initialization,
                "model": model.state_dict(), "target_model": target_model.state_dict(), "optimizer": optimizer.state_dict(),
                "replay_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if resolved.type == "cuda" else [],
                "device": str(resolved),
            }
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
    return path
