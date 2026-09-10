from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from intrep.experience.multimodal.records import (
    MultimodalEpisode,
    save_episode,
    write_audio,
    write_image,
)
from intrep.representation.assemblies.multimodal_agent import (
    MultimodalAgentModel,
    PredictedOutcome,
)
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.worlds.gridworld.multimodal import (
    SAMPLE_RATE,
    MultimodalNavigationWorld,
)
from intrep.worlds.gridworld.world import GRID_ACTIONS


@dataclass
class AgentDecision:
    action: int
    action_values: torch.Tensor
    text: str
    prediction: PredictedOutcome


class AgentSession:
    """Streaming inference memory is independent of training replay storage."""

    def __init__(self, model: MultimodalAgentModel, *, checkpoint_id: str, seed: int = 0) -> None:
        self.model = model.eval()
        self.checkpoint_id = checkpoint_id
        self.generator = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.memory = self.model.new_memory(1).detach().clone()
        self.step = 0

    @torch.no_grad()
    def act(self, observation: MultimodalObservation, *, epsilon: float = 0.0, max_text_bytes: int = 16) -> AgentDecision:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        self.memory = self.model.observe([observation], self.memory, step=self.step).detach()
        values = self.model.policy(self.memory)[0]
        if float(torch.rand((), generator=self.generator)) < epsilon:
            action = int(torch.randint(self.model.config.action_count, (), generator=self.generator))
        else:
            action = int(values.argmax())
        shape = None if observation.image is None else tuple(observation.image.shape[:2])
        samples = 0 if observation.audio is None else len(observation.audio)
        prediction = self.model.predict_outcome(self.memory, torch.tensor([action], device=self.memory.device),
                                               image_shapes=[shape], audio_samples=samples,
                                               sample_rate=observation.sample_rate)
        text = self.model.generate_text(self.memory, max_bytes=max_text_bytes)[0]
        self.step += 1
        return AgentDecision(action, values.cpu(), text, prediction)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"schema_version": "intrep.multimodal_session.v1", "checkpoint_id": self.checkpoint_id,
                    "step": self.step, "memory": self.memory.cpu(), "rng": self.generator.get_state()}, temporary)
        temporary.replace(path)

    def restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != "intrep.multimodal_session.v1" or payload["checkpoint_id"] != self.checkpoint_id:
            raise ValueError("session belongs to a different model checkpoint")
        memory = payload["memory"].to(self.memory.device)
        self.model._memory(memory)
        if len(memory) != 1 or payload["step"] < 0 or not torch.isfinite(memory).all():
            raise ValueError("invalid inference session state")
        self.memory, self.step = memory, payload["step"]
        self.generator.set_state(payload["rng"])


def checkpoint_identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_decision(directory: Path, decision: AgentDecision, *, sample_rate: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    row = {"action": decision.action, "action_values": decision.action_values.tolist(), "text": decision.text,
           "predicted_reward": float(decision.prediction.feedback[0, 0]),
           "predicted_termination_probability": float(decision.prediction.feedback[0, 1].sigmoid()),
           "predicted_truncation_probability": float(decision.prediction.feedback[0, 2].sigmoid())}
    if decision.prediction.images[0] is not None:
        write_image(directory / "predicted.png", decision.prediction.images[0])
        row["predicted_image"] = "predicted.png"
    if decision.prediction.audio.shape[1]:
        write_audio(directory / "predicted.wav", decision.prediction.audio[0], sample_rate)
        row["predicted_audio"] = "predicted.wav"
    (directory / "decision.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return row


@torch.no_grad()
def rollout(
    model: MultimodalAgentModel, *, checkpoint_id: str, seed: int, source_root: Path | None = None,
    display_root: Path | None = None, horizon: int = 8, epsilon: float = 0.0,
) -> tuple[MultimodalEpisode, dict]:
    if model.config.action_count != len(GRID_ACTIONS):
        raise ValueError("navigation rollout requires its five-action vocabulary")
    world = MultimodalNavigationWorld(seed, horizon=horizon)
    session = AgentSession(model, checkpoint_id=checkpoint_id, seed=seed)
    observations = [world.observe()]
    actions, traces = [], []
    for step in range(horizon):
        decision = session.act(observations[-1], epsilon=epsilon)
        # Evaluation annotations are computed after the actor fixes its decision.
        expected_text = ("red", "blue")[world.target]
        teacher = world.expert_action()
        before = world.world.hidden_state
        target_position = before.goal
        next_observation = world.step(decision.action)
        after = world.world.hidden_state
        distance = abs(after.agent.row - target_position.row) + abs(after.agent.col - target_position.col)
        actions.append(decision.action)
        row = {"step": step, "action": decision.action, "action_name": GRID_ACTIONS[decision.action],
               "action_values": decision.action_values.tolist(), "text": decision.text,
               "expected_text": expected_text, "teacher_action": teacher,
               "reward": float(next_observation.feedback[0]), "at_target": distance == 0,
               "distance_to_target": distance,
               "image_mse": float((decision.prediction.images[0] - next_observation.image.to(session.memory.device)).square().mean()),
               "audio_mse": float((decision.prediction.audio[0] - next_observation.audio.to(session.memory.device)).square().mean()),
               "predicted_reward": float(decision.prediction.feedback[0, 0])}
        if display_root is not None:
            directory = display_root / f"step-{step:03d}"
            save_decision(directory, decision, sample_rate=SAMPLE_RATE)
            write_image(directory / "observed.png", observations[-1].image)
            write_image(directory / "actual.png", next_observation.image)
            write_audio(directory / "heard.wav", observations[-1].audio, SAMPLE_RATE)
            write_audio(directory / "actual.wav", next_observation.audio, SAMPLE_RATE)
            row["input_text"] = observations[-1].text
        observations.append(next_observation)
        traces.append(row)
    episode_id = f"actor-{checkpoint_id[:12]}-{seed}-{horizon}"
    # Privileged evaluation annotations stay in the report, not the replay targets.
    episode = MultimodalEpisode(episode_id, world.world_id, observations, actions,
                               provenance={"actor_checkpoint": checkpoint_id, "seed": seed, "epsilon": epsilon,
                                           "generator": "multimodal_navigation", "horizon": horizon})
    if source_root is not None:
        source_path = save_episode(source_root, episode)
    else:
        source_path = None
    summary = {"episode_id": episode_id, "world_id": world.world_id,
               "source_path": str(source_path) if source_path else None, "transitions": traces,
               "total_reward": sum(row["reward"] for row in traces),
               "target_steps": sum(row["at_target"] for row in traces),
               "text_accuracy": sum(row["text"] == row["expected_text"] for row in traces) / horizon}
    if display_root is not None:
        (display_root / "rollout.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        session.save(display_root / "session.pt")
    return episode, summary
