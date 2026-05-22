# Shogi Status

Last updated: 2026-05-17.

This is the compact current-state document. Detailed experiment rows live in
`learning-experiments.md`; runtime measurements live in the throughput and
inference-performance docs.

## Current Promoted Checkpoint

No current promoted component checkpoint is stored in `models/`.

Historical measurements may refer to `d256-h1024-heads8-l6-shogi`, which was
the previous single-file promoted checkpoint before checkpoint artifacts were
split into input, core, policy output, and value output components.

## Training Data

Current successful full training uses the Qhapaq full tensor cache:

- train examples: 4,951,012
- eval examples: 262,133
- cache size: about 27 GB

## Latest Learning Result

The latest online replay run trained for 4 iterations from the Qhapaq full-cache
checkpoint. The final iteration improved fixed-eval policy/value metrics
slightly:

- eval loss: 1.8461 -> 1.8436
- eval accuracy: 0.4495 -> 0.4512

## Latest Playing-Strength Check

Against the previous promoted checkpoint, with sampled checkpoint move selection,
100 games, alternating sides, MCTS128, and batch64:

- result: 53-39-8
- average plies: 174.95
