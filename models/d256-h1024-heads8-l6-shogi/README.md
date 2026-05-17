# d256-h1024-heads8-l6-shogi

This is the current promoted shogi policy/value checkpoint.

Promoted from:

```text
shogi-learning-20260517-001
```

Artifact:

```text
models/d256-h1024-heads8-l6-shogi/checkpoint.pt
```

## Model

- architecture: d256-h1024-heads8-l6
- problem: shogi policy/value
- trained objective in this run: policy + winner value

## Training Data

- source: Qhapaq full tensor cache
- train examples: 4,951,012
- eval examples: 262,133
- train eval cap during training: 65,536
- final eval cap during training: full eval set

## Training Result

- train loss: 2.0753 -> 1.9431
- train value loss: 1.0151 -> 0.9064
- eval loss: 2.0960 -> 1.9751
- eval value loss: 1.0150 -> 0.9084
- eval accuracy: 0.4059 -> 0.4257
- eval top-3 accuracy: 0.6818
- eval top-5 accuracy: 0.7795

## Playing-Strength Checks

Sampled checkpoint move selection, alternating sides, MCTS128, batch64:

- vs previous promoted checkpoint: 14-1-1

Evaluation move selection, alternating sides, MCTS128, batch64:

- vs YaneuraOu MaterialLv1 `go nodes 1`: 10-6

Same-checkpoint self-play with sampled checkpoint move selection, alternating
sides, MCTS128, batch64:

- player A perspective: 4-11-1
- illegal moves: 0

## Integrity

SHA256:

```text
4057ac647b505d632653d5e9f9b16fbae4d730751125a9f5004d20bd6c9e29bd
```
