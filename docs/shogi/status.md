# Shogi Status

Last updated: 2026-05-17.

This is the compact current-state document. Detailed experiment rows live in
`learning-experiments.md`; runtime measurements live in the throughput and
inference-performance docs.

## Current Model

The strongest project checkpoint observed so far is the one-epoch Qhapaq
full-cache policy/value checkpoint from `shogi-learning-20260517-001`.

It is promoted here:

```text
models/d256-h1024-heads8-l6-shogi/checkpoint.pt
```

## Training Data

Current successful full training uses the Qhapaq full tensor cache:

- train examples: 4,951,012
- eval examples: 262,133
- cache size: about 27 GB

## Latest Learning Result

One full Qhapaq policy/value epoch trained cleanly:

- train loss: 2.0753 -> 1.9431
- train value loss: 1.0151 -> 0.9064
- eval loss: 2.0960 -> 1.9751
- eval value loss: 1.0150 -> 0.9084
- eval accuracy: 0.4059 -> 0.4257
- eval top-3 accuracy: 0.6818
- eval top-5 accuracy: 0.7795

## Latest Playing-Strength Check

Against the previous promoted checkpoint, with sampled checkpoint move
selection, alternating sides, MCTS128, and batch64:

- result: 14-1-1
- average plies: 136.56

Against YaneuraOu MaterialLv1 `go nodes 1`, with evaluation move selection,
alternating sides, MCTS128, and batch64:

- result: 10-6
- average plies: 111.25

In same-checkpoint self-play, with sampled checkpoint move selection,
alternating sides, MCTS128, and batch64:

- result: 4-11-1 from player A's perspective
- average plies: 169.56
