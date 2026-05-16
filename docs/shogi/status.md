# Shogi Status

Last updated: 2026-05-16.

This is the compact current-state document. Detailed experiment rows live in
`learning-experiments.md`; runtime measurements live in the throughput and
inference-performance docs.

## Current Model

The strongest measured project checkpoint is the one-epoch Qhapaq full-cache
policy checkpoint from `shogi-learning-20260516-003`.

It is still a run artifact path, not an immutable promoted model identity:

```text
runs/shogi/runpod-qhapaq-full-one-epoch-rtx4090-20260516/checkpoint.pt
```

## Training Data

Current successful full training uses the Qhapaq full tensor cache:

- train examples: 4,951,012
- eval examples: 262,133
- skipped games: 2
- cache size: about 27 GB

## Latest Learning Result

One full Qhapaq epoch trained cleanly:

- train loss: 4.2407 -> 2.0753
- eval loss: 4.2447 -> 2.1020
- eval accuracy: 0.0259 -> 0.4039
- eval top-3 accuracy: 0.6540
- eval top-5 accuracy: 0.7522

## Latest Playing-Strength Check

With sampled checkpoint move selection, alternating sides, MCTS128, and batch64:

- vs starting checkpoint: 16-0
- vs YaneuraOu MaterialLv1 `go nodes 1`: 0-16
- no draws or illegal moves in either 16-game match
- every game in each match had a unique move sequence

Current interpretation: the trained checkpoint is clearly above the starting
checkpoint under these settings, but still below YaneuraOu MaterialLv1 nodes1.

## Current Compute Shape

RunPod official PyTorch 2.8 template is the working GPU environment. The most
recent successful match run used RTX A5000, 9 vCPU, 50 GB RAM, no network
volume.

For full Qhapaq tensor-cache training, `num_workers=8` was the fastest measured
worker count among 0/2/4/8 on the measured A40 pod.

## Known Constraints

- `runs/` is disposable and must not be the canonical home for promoted models.
- Checkpoint identity and promotion are not yet settled.
- Self-play and MCTS-heavy generation remain CPU-sensitive.
- Playing-strength evidence should come from game-record JSONL; docs only keep
  compact summaries.

## Next Useful Work

1. Decide how to promote the trained checkpoint into a durable model identity.
2. Run a stronger external comparison after promotion, not against a run path.
3. Continue training from the current strongest checkpoint with more data or
   additional epochs.
