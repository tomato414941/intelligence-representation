"""Measure a frozen agent on identical histories with individual input perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from intrep.experience.multimodal.records import selected_episodes
from intrep.problems.multimodal_agent.runtime import checkpoint_identity
from intrep.problems.multimodal_agent.training import load_checkpoint


@torch.no_grad()
def evaluate(checkpoint: Path, selection: Path, split: str, device: str, batch_size: int) -> dict:
    model, payload = load_checkpoint(checkpoint, device=device)
    model.eval()
    episodes = selected_episodes(selection, split)
    trained = {world for source in payload.get("training_history", payload["sources"]) for world in source["world_ids"]}
    if not episodes or trained.intersection(episode.world_id for episode in episodes):
        raise ValueError("diagnostics require nonempty, unseen evaluation worlds")
    conditions = ("full", "empty_text", "blank_image", "silent_audio", "reset_memory")
    results = {}
    for condition in conditions:
        rows = []
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start:start + batch_size]
            memory = model.new_memory(len(batch))
            for step in range(max(len(episode.actions) for episode in batch)):
                active = [index for index, episode in enumerate(batch) if step < len(episode.actions)]
                indices = torch.tensor(active, device=memory.device)
                observations = [batch[index].observations[step] for index in active]
                if condition == "empty_text":
                    observations = [replace(row, text="") for row in observations]
                elif condition == "blank_image":
                    observations = [replace(row, image=torch.zeros_like(row.image) if row.image is not None else None)
                                    for row in observations]
                elif condition == "silent_audio":
                    observations = [replace(row, audio=torch.zeros_like(row.audio) if row.audio is not None else None)
                                    for row in observations]
                previous = model.new_memory(len(active)) if condition == "reset_memory" else memory[indices]
                updated = model.observe(observations, previous, step=step)
                memory = memory.index_copy(0, indices, updated)
                actions = model.policy(updated).argmax(-1).tolist()
                texts = model.generate_text(updated, max_bytes=16)
                forecasts = {}
                if condition == "full":
                    next_observations = [batch[index].observations[step + 1] for index in active]
                    for rate in sorted({row.sample_rate for row in next_observations}):
                        locals_ = [local for local, row in enumerate(next_observations) if row.sample_rate == rate]
                        following = [next_observations[local] for local in locals_]
                        prediction = model.predict_outcome(
                            updated[locals_],
                            torch.tensor([batch[active[local]].actions[step] for local in locals_], device=memory.device),
                            image_shapes=[tuple(row.image.shape[:2]) if row.image is not None else None for row in following],
                            audio_samples=max(len(row.audio) if row.audio is not None else 0 for row in following),
                            sample_rate=rate,
                        )
                        for output_index, local in enumerate(locals_):
                            future, previous = next_observations[local], observations[local]
                            scores = {"forecast_reward_mse": float((prediction.feedback[output_index, 0]
                                                                   - future.feedback[0].to(memory.device)).square())}
                            if future.image is not None:
                                scores["forecast_image_mse"] = float((prediction.images[output_index]
                                                                      - future.image.to(memory.device)).square().mean())
                                if previous.image is not None and previous.image.shape == future.image.shape:
                                    scores["copy_image_mse"] = float((previous.image - future.image).square().mean())
                            if future.audio is not None:
                                scores["forecast_audio_mse"] = float((prediction.audio[output_index, :len(future.audio)]
                                                                      - future.audio.to(memory.device)).square().mean())
                                scores["silence_audio_mse"] = float(future.audio.square().mean())
                            forecasts[local] = scores
                for local, (index, action, text) in enumerate(zip(active, actions, texts)):
                    episode = batch[index]
                    teacher = episode.teacher_actions[step] if episode.teacher_actions else None
                    answer = episode.answers[step] if episode.answers else None
                    rows.append({"id": episode.id, "step": step, "action": action, "text": text,
                                 "teacher_action": teacher, "answer": answer,
                                 "action_match": action == teacher if teacher is not None else None,
                                 "text_match": text == answer if answer is not None else None,
                                 **forecasts.get(local, {})})

        def score(group: list[dict]) -> dict:
            actions = [row["action_match"] for row in group if row["action_match"] is not None]
            texts = [row["text_match"] for row in group if row["text_match"] is not None]
            return {"teacher_action_accuracy": sum(actions) / len(actions) if actions else None,
                    "text_accuracy": sum(texts) / len(texts) if texts else None,
                    "action_count": len(actions), "text_count": len(texts)}

        results[condition] = {**score(rows), "by_step": {
            str(step): score([row for row in rows if row["step"] == step])
            for step in sorted({row["step"] for row in rows})}, "rows": rows}
        print(json.dumps({"condition": condition, **score(rows)}), flush=True)
    full = results["full"]["rows"]
    action_counts = Counter(row["teacher_action"] for row in full if row["teacher_action"] is not None)
    text_counts = Counter(row["answer"] for row in full if row["answer"] is not None)
    forecast_means = {}
    for name in ("forecast_image_mse", "copy_image_mse", "forecast_audio_mse", "silence_audio_mse", "forecast_reward_mse"):
        values = [row[name] for row in full if name in row]
        if values:
            forecast_means[name] = sum(values) / len(values)
    return {"checkpoint_sha256": checkpoint_identity(checkpoint), "checkpoint_step": payload["step"],
            "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            "split": split, "episode_count": len(episodes), "conditions": results,
            "best_constant_action_accuracy": max(action_counts.values()) / sum(action_counts.values()) if action_counts else None,
            "majority_text_accuracy": max(text_counts.values()) / sum(text_counts.values()) if text_counts else None,
            "forecast_means": forecast_means,
            "interpretation": "Frozen-model input perturbations on recorded histories; distribution shift is included. "
                              "Teacher agreement counts one labeled action, including arbitrary tie breaking. "
                              "These measurements do not establish performance after retraining without a modality."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    result = evaluate(args.checkpoint, args.selection, args.split, args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
