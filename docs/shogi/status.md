# Shogi Status

Last updated: 2026-05-16.

This is the compact current-state document. Detailed experiment rows live in
`learning-experiments.md`; runtime measurements live in the throughput and
inference-performance docs.

## Current Model

The strongest project checkpoint observed so far is the one-epoch Qhapaq
full-cache policy checkpoint from `shogi-learning-20260516-003`.

It is promoted here:

```text
models/d256-h1024-heads8-l6-shogi/checkpoint.pt
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

Against YaneuraOu MaterialLv1 `go nodes 1`, with evaluation move selection,
alternating sides, and batch64:

- MCTS128: 2-2
- MCTS256: 3-1
- MCTS512: 1-3
- MCTS1024: 0-4
- MCTS2048: 0-4
- 4 games per MCTS case

Current interpretation: against YaneuraOu MaterialLv1 nodes1, the checkpoint can
take games around MCTS128-256 in a small sample, but the result is not yet
robust.
