from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from intrep.experience.multimodal.records import (
    read_audio,
    save_episode,
    selected_episodes,
    write_selection,
)
from intrep.problems.multimodal_agent.online import collect_and_learn
from intrep.problems.multimodal_agent.runtime import (
    AgentSession,
    checkpoint_identity,
    rollout,
    save_decision,
)
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    episode_loss,
    load_checkpoint,
    train,
)
from intrep.representation.assemblies.multimodal_agent import MultimodalAgentConfig
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.worlds.gridworld.multimodal import generate_episode


def prepare(root: Path, *, train_count: int, validation_count: int, test_count: int, seed: int, horizon: int) -> Path:
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("all split sizes must be positive")
    if (root / "selection.json").exists():
        raise FileExistsError("data selection already exists")
    partitions: dict[str, list[Path]] = {}
    seen_worlds: set[str] = set()
    candidate_seed = seed
    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        paths = []
        while len(paths) < count:
            episode = generate_episode(candidate_seed, horizon=horizon)
            candidate_seed += 1
            if episode.world_id in seen_worlds:
                continue
            seen_worlds.add(episode.world_id)
            paths.append(save_episode(root / "episodes", episode))
        partitions[split] = paths
    return write_selection(root, partitions)


@torch.no_grad()
def evaluate(checkpoint: Path, selection: Path, *, split: str, device: str, limit: int | None = None) -> dict:
    model, payload = load_checkpoint(checkpoint, device=device)
    model.eval()
    episodes = selected_episodes(selection, split)
    if limit is not None:
        if limit < 1:
            raise ValueError("evaluation limit must be positive")
        episodes = episodes[:limit]
    training_history = payload.get("training_history", payload["sources"])
    training_ids = {name for source in training_history for name in source.get("episode_ids", [])}
    training_worlds = {name for source in training_history for name in source.get("world_ids", [])}
    if split != "train" and (training_ids.intersection(episode.id for episode in episodes)
                             or training_worlds.intersection(episode.world_id for episode in episodes)):
        raise ValueError("evaluation episodes or worlds were included in training")
    config = MultimodalTrainingConfig(**{**payload["config"], "model": model.config})
    rows = []
    for episode in episodes:
        loss, components = episode_loss(model, [episode], config)
        memory = model.new_memory()
        correct, total, text_correct, text_total = 0, 0, 0, 0
        for step, observation in enumerate(episode.observations[:-1]):
            memory = model.observe([observation], memory, step=step)
            if episode.teacher_actions and episode.teacher_actions[step] is not None:
                correct += int(model.policy(memory)[0].argmax()) == episode.teacher_actions[step]
                total += 1
            if episode.answers and episode.answers[step] is not None:
                text_correct += model.generate_text(memory, max_bytes=16)[0] == episode.answers[step]
                text_total += 1
        rows.append({"id": episode.id, "loss": float(loss), **components,
                     "teacher_matches": correct, "action_count": total, "text_matches": text_correct, "text_count": text_total})
    action_count, text_count = sum(row["action_count"] for row in rows), sum(row["text_count"] for row in rows)
    return {"checkpoint_sha256": checkpoint_identity(checkpoint), "checkpoint_step": payload["step"], "split": split,
            "episode_count": len(rows), "episodes": rows,
            "teacher_action_accuracy": sum(row["teacher_matches"] for row in rows) / action_count if action_count else None,
            "text_accuracy": sum(row["text_matches"] for row in rows) / text_count if text_count else None,
            "mean_losses": {name: sum(row[name] for row in rows) / len(rows)
                            for name in ("loss", "teacher", "text", "image", "audio", "feedback")}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated multimodal predictive agent")
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--train-count", type=int, default=1024)
    data.add_argument("--validation-count", type=int, default=64)
    data.add_argument("--test-count", type=int, default=128)
    data.add_argument("--seed", type=int, default=82001)
    data.add_argument("--horizon", type=int, default=6)
    learning = sub.add_parser("train")
    learning.add_argument("--selection", type=Path, action="append", required=True)
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--steps", type=int, default=3000)
    learning.add_argument("--batch-size", type=int, default=16)
    learning.add_argument("--replay-capacity", type=int, default=4096)
    learning.add_argument("--learning-rate", type=float, default=0.0003)
    learning.add_argument("--seed", type=int, default=41)
    learning.add_argument("--resume", action="store_true")
    learning.add_argument("--initialize", type=Path)
    learning.add_argument("--model-config", type=Path)
    learning.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--selection", type=Path, required=True)
    evaluation.add_argument("--split", default="test")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    actor = sub.add_parser("rollout")
    actor.add_argument("--checkpoint", type=Path, required=True)
    actor.add_argument("--output", type=Path, required=True)
    actor.add_argument("--seed", type=int, default=94001)
    actor.add_argument("--episodes", type=int, default=8)
    actor.add_argument("--horizon", type=int, default=8)
    actor.add_argument("--epsilon", type=float, default=0.0)
    actor.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--text", default="")
    inference.add_argument("--image", type=Path)
    inference.add_argument("--audio", type=Path)
    inference.add_argument("--previous-action", type=int)
    inference.add_argument("--feedback", type=float, nargs=3, metavar=("REWARD", "TERMINATED", "TRUNCATED"))
    inference.add_argument("--session", type=Path)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--max-text-bytes", type=int, default=128)
    inference.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    cycle = sub.add_parser("cycle")
    cycle.add_argument("--checkpoint", type=Path, required=True)
    cycle.add_argument("--selection", type=Path, action="append", required=True)
    cycle.add_argument("--output", type=Path, required=True)
    cycle.add_argument("--rounds", type=int, default=1)
    cycle.add_argument("--episodes", type=int, default=16)
    cycle.add_argument("--learn-steps", type=int, default=100)
    cycle.add_argument("--horizon", type=int, default=8)
    cycle.add_argument("--seed", type=int, default=95001)
    cycle.add_argument("--epsilon", type=float, default=0.15)
    cycle.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        print(prepare(args.output, train_count=args.train_count, validation_count=args.validation_count,
                      test_count=args.test_count, seed=args.seed, horizon=args.horizon))
    elif args.command == "train":
        model = MultimodalAgentConfig(**json.loads(args.model_config.read_text())) if args.model_config else MultimodalAgentConfig()
        config = MultimodalTrainingConfig(model=model, steps=args.steps, batch_size=args.batch_size,
                                         replay_capacity=args.replay_capacity, learning_rate=args.learning_rate, seed=args.seed)
        print(train(args.selection, args.output, config, device=args.device, resume=args.resume, initialize=args.initialize))
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.selection, split=args.split, device=args.device, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
    elif args.command == "rollout":
        if args.episodes < 1:
            parser.error("episodes must be positive")
        model, _ = load_checkpoint(args.checkpoint, device=args.device)
        identity = checkpoint_identity(args.checkpoint)
        sources, reports = [], []
        for index in range(args.episodes):
            _, report = rollout(model, checkpoint_id=identity, seed=args.seed + index,
                                source_root=args.output / "episodes", display_root=args.output / f"replay-{index:03d}",
                                horizon=args.horizon, epsilon=args.epsilon)
            sources.append(Path(report["source_path"]))
            reports.append(report)
        selection = write_selection(args.output, {"train": sources})
        result = {"checkpoint_sha256": identity, "episodes": reports, "replay_selection": str(selection)}
        (args.output / "rollouts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"replay_selection": str(selection), "episodes": len(reports),
                          "mean_reward": sum(row["total_reward"] for row in reports) / len(reports)}))
    elif args.command == "infer":
        model, _ = load_checkpoint(args.checkpoint, device=args.device)
        session = AgentSession(model, checkpoint_id=checkpoint_identity(args.checkpoint))
        if args.session:
            session.restore(args.session)
        image = audio = None
        rate = 16000
        if args.image:
            with Image.open(args.image) as source:
                image = torch.from_numpy(np.array(source.convert("RGB"), dtype=np.float32) / 255)
        if args.audio:
            audio, rate = read_audio(args.audio)
        feedback = torch.tensor(args.feedback) if args.feedback is not None else None
        observation = MultimodalObservation(args.text, image, audio, rate, args.previous_action, feedback)
        decision = session.act(observation, max_text_bytes=args.max_text_bytes)
        print(json.dumps(save_decision(args.output, decision, sample_rate=rate), ensure_ascii=False))
        session.save(args.output / "session.pt")
    elif args.command == "cycle":
        print(collect_and_learn(args.checkpoint, args.selection, args.output, rounds=args.rounds,
                               episodes_per_round=args.episodes, learning_steps=args.learn_steps,
                               horizon=args.horizon, seed=args.seed, epsilon=args.epsilon, device=args.device))


if __name__ == "__main__":
    main()
