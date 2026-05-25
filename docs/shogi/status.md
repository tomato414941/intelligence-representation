# Shogi Status

Last updated: 2026-05-25.

This is the compact current-state document. Runtime measurements live in the
throughput and inference-performance docs. Playing-strength results live in
`docs/shogi/playing-strength.md`.

## Shogi Checkpoints Under Evaluation

Loadable component checkpoints currently used for shogi entry comparison:

- minimal single-global action-plane:
  `models/shogi-minimal-single-global-position-action-plane/`

- minimal split-global action-plane:
  `models/shogi-minimal-split-global-action-plane/`

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

## Current Playing-Strength Readout

See `docs/shogi/playing-strength.md` for match results. Current readout:

- minimal single-global lost to Suisho5 `go nodes 1000` by 4-12-0.
- minimal split-global beat minimal single-global by 8-7-1 in the latest
  seeded-opening entry comparison.

## Older Lessons

- Small deterministic matches can repeat the same game lines. Sampled move
  selection removed that artifact in the 2026-05-16 checks.
- The 2026-05-17 Qhapaq policy/value winner-training run improved fixed eval
  metrics, beat the previous promoted checkpoint 14-1-1, beat YaneuraOu
  MaterialLv1 nodes1/10/100 in 16-game checks, and lost to nodes1000 by 37-61-2
  over 100 games.
