from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from intrep.representation.inputs.multimodal_observation import MultimodalObservation

EPISODE_SCHEMA = "intrep.multimodal_episode.v1"
SELECTION_SCHEMA = "intrep.multimodal_selection.v1"


def read_audio(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("expected uncompressed 16-bit PCM WAV audio")
        channels, rate = source.getnchannels(), source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()
    audio = samples.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
    return torch.from_numpy(audio), rate


def write_audio(path: Path, samples: torch.Tensor, sample_rate: int) -> None:
    if samples.ndim != 1 or not samples.numel() or not torch.isfinite(samples).all() or sample_rate < 1:
        raise ValueError("audio must be nonempty finite mono audio with positive sample rate")
    pcm = (samples.detach().cpu().clamp(-1, 1).numpy() * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())


def write_image(path: Path, image: torch.Tensor) -> None:
    if image.ndim != 3 or image.shape[-1] != 3 or not torch.isfinite(image).all():
        raise ValueError("image must be a finite HWC RGB tensor")
    pixels = (image.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _relative_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("media path escapes its episode directory")
    return path


@dataclass
class MultimodalEpisode:
    id: str
    world_id: str
    observations: list[MultimodalObservation]
    actions: list[int]
    teacher_actions: list[int | None] = field(default_factory=list)
    answers: list[str | None] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id or Path(self.id).name != self.id or self.id in {".", ".."} or not self.world_id:
            raise ValueError("episode requires a simple id and a world identity")
        if not self.actions or len(self.observations) != len(self.actions) + 1:
            raise ValueError("an episode must have one more observation than executed actions")
        for labels in (self.teacher_actions, self.answers):
            if labels and len(labels) != len(self.actions):
                raise ValueError("optional targets must align with executed transitions")
        if self.observations[0].previous_action is not None or self.observations[0].feedback is not None:
            raise ValueError("initial observation must not carry an earlier episode's action or feedback")
        for index, action in enumerate(self.actions):
            observation = self.observations[index + 1]
            if action < 0 or observation.previous_action != action or observation.feedback is None:
                raise ValueError("executed action and following observation must agree")
            if observation.feedback.shape != (3,) or not torch.isfinite(observation.feedback).all():
                raise ValueError("feedback must contain finite reward, terminated and truncated values")
            if not torch.all((observation.feedback[1:] == 0) | (observation.feedback[1:] == 1)):
                raise ValueError("termination and truncation flags must be binary")
            if index < len(self.actions) - 1 and observation.feedback[1:].any():
                raise ValueError("episode continues after termination or truncation")


def save_episode(root: Path, episode: MultimodalEpisode) -> Path:
    episode.validate()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / episode.id
    if destination.exists():
        raise FileExistsError("source episode already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".episode-", dir=root))
    try:
        observations = []
        for index, observation in enumerate(episode.observations):
            row: dict = {"text": observation.text, "previous_action": observation.previous_action}
            if observation.feedback is not None:
                row["feedback"] = observation.feedback.detach().cpu().tolist()
            if observation.image is not None:
                row["image"] = f"observation-{index:04d}.png"
                write_image(temporary / row["image"], observation.image)
            if observation.audio is not None:
                row["audio"] = f"observation-{index:04d}.wav"
                write_audio(temporary / row["audio"], observation.audio, observation.sample_rate)
            observations.append(row)
        payload = {
            "schema_version": EPISODE_SCHEMA, "id": episode.id, "world_id": episode.world_id,
            "observations": observations, "actions": episode.actions,
            "targets": {"teacher_actions": episode.teacher_actions, "answers": episode.answers},
            "provenance": episode.provenance,
        }
        (temporary / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "episode.json"


def load_episode(path: Path) -> MultimodalEpisode:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != EPISODE_SCHEMA:
        raise ValueError("unsupported multimodal episode schema")
    observations = []
    for row in payload["observations"]:
        image = audio = None
        sample_rate = 16000
        if row.get("image"):
            with Image.open(_relative_file(path.parent, row["image"])) as source:
                image = torch.from_numpy(np.array(source.convert("RGB"), dtype=np.float32) / 255)
        if row.get("audio"):
            audio, sample_rate = read_audio(_relative_file(path.parent, row["audio"]))
        feedback = None if "feedback" not in row else torch.tensor(row["feedback"], dtype=torch.float32)
        observations.append(MultimodalObservation(row.get("text", ""), image, audio, sample_rate,
                                                  row.get("previous_action"), feedback))
    targets = payload.get("targets", {})
    episode = MultimodalEpisode(payload["id"], payload["world_id"], observations, payload["actions"],
                               targets.get("teacher_actions", []), targets.get("answers", []),
                               payload.get("provenance", {}))
    episode.validate()
    return episode


def episode_digest(path: Path) -> str:
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes())
    for observation in payload["observations"]:
        for kind in ("image", "audio"):
            if observation.get(kind):
                digest.update(_relative_file(path.parent, observation[kind]).read_bytes())
    return digest.hexdigest()


def write_selection(root: Path, partitions: dict[str, list[Path]], *, filename: str = "selection.json") -> Path:
    seen_ids: set[str] = set()
    world_splits: dict[str, str] = {}
    entries = []
    for split, paths in partitions.items():
        for path in paths:
            payload = json.loads(path.read_text())
            episode_id, world_id = payload["id"], payload["world_id"]
            if episode_id in seen_ids or world_splits.get(world_id, split) != split:
                raise ValueError("episode duplication or world leakage across data splits")
            seen_ids.add(episode_id)
            world_splits[world_id] = split
            relative = path.resolve().relative_to(root.resolve())
            entries.append({"id": episode_id, "world_id": world_id, "split": split,
                            "path": relative.as_posix(), "sha256": episode_digest(path)})
    target = root / filename
    if target.exists():
        raise FileExistsError("selection already exists; create an explicit new version")
    target.write_text(json.dumps({"schema_version": SELECTION_SCHEMA, "episodes": entries}, indent=2) + "\n")
    return target


def selected_episodes(selection: Path, split: str) -> list[MultimodalEpisode]:
    payload = json.loads(selection.read_text())
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported multimodal selection schema")
    episodes = []
    seen_ids: set[str] = set()
    world_splits: dict[str, str] = {}
    for row in payload["episodes"]:
        if row["id"] in seen_ids or world_splits.get(row["world_id"], row["split"]) != row["split"]:
            raise ValueError("selection duplicates episodes or crosses world splits")
        seen_ids.add(row["id"])
        world_splits[row["world_id"]] = row["split"]
        if row["split"] != split:
            continue
        path = _relative_file(selection.parent, row["path"])
        if episode_digest(path) != row["sha256"]:
            raise ValueError("source episode changed after data selection")
        episode = load_episode(path)
        if episode.id != row["id"] or episode.world_id != row["world_id"]:
            raise ValueError("selected episode identity does not match source")
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"selection has no episodes for split {split!r}")
    return episodes
