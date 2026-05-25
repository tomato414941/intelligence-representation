# Shogi Status

Last updated: 2026-05-25.

This is the compact current-state document. Runtime measurements live in the
throughput and inference-performance docs.

## Current Promoted Checkpoint

The current loadable component checkpoint stored in `models/` is:

```text
models/shogi-minimal-single-global-position-action-plane/
```

It is the latest full Qhapaq-trained minimal single-global action-plane
checkpoint.

## Training Data

Current successful full training uses the Qhapaq full tensor cache:

- train examples: 4,951,012
- eval examples: 262,133
- cache size: about 27 GB

## Latest Learning Result

The latest full Qhapaq tensor-cache run trained the minimal single-global
action-plane entry and stopped early:

- actual steps: 68,000
- best eval step: 58,000
- best eval loss: 1.8352
- final eval accuracy: 0.4543
- final eval value loss: 0.8854

## Latest Playing-Strength Check

Against Suisho5 `go nodes 1000`, with alternating sides, MCTS128, and NN leaf
eval batch limit 64:

- result: 4-12-0
- illegal moves: 0
- average plies: 107.375
- side split: black 1-7, white 3-5

## Latest Entry Comparison

Minimal split-global action-plane vs minimal single-global action-plane, using
8 seeded random-legal opening positions with both side assignments, MCTS128, and
NN leaf eval batch limit 64:

- result: split-global 8, single-global 7, draws 1
- illegal moves: 0
- average plies: 192.0625
- split-global side split: black 5-3, white 3-4
- start positions: seed 20260525, opening plies 8

## Previous Playing-Strength Check

Against YaneuraOu MaterialLv1 `go nodes 1000`, with alternating sides, MCTS128,
and NN leaf eval batch limit 64:

- result: 16-0-0
- illegal moves: 0
- average plies: 157.25

Against Suisho5 `go nodes 1`, with alternating sides, MCTS128, and NN leaf eval
batch limit 64:

- result: 9-7-0
- illegal moves: 0
- average plies: 180.4375
- side split: black 7-1, white 2-6

Against Suisho5 `go nodes 1000`, with alternating sides, MCTS128, and NN leaf
eval batch limit 64:

- result: 7-9-0
- illegal moves: 0
- average plies: 141.1875
- side split: black 3-5, white 4-4

## Older Lessons

- Small deterministic matches can repeat the same game lines. Sampled move
  selection removed that artifact in the 2026-05-16 checks.
- The 2026-05-17 Qhapaq policy/value winner-training run improved fixed eval
  metrics, beat the previous promoted checkpoint 14-1-1, beat YaneuraOu
  MaterialLv1 nodes1/10/100 in 16-game checks, and lost to nodes1000 by 37-61-2
  over 100 games.
