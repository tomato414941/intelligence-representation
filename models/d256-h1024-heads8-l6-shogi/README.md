# d256-h1024-heads8-l6-shogi

This is the current promoted shogi policy checkpoint.

Promoted from:

```text
shogi-learning-20260516-003
```

Artifact:

```text
models/d256-h1024-heads8-l6-shogi/checkpoint.pt
```

## Model

- architecture: d256-h1024-heads8-l6
- problem: shogi policy/value
- trained objective in this run: policy-only

## Training Data

- source: Qhapaq full tensor cache
- train examples: 4,951,012
- eval examples: 262,133
- eval cap during training: 65,536

## Training Result

- train loss: 4.2407 -> 2.0753
- eval loss: 4.2447 -> 2.1020
- eval accuracy: 0.0259 -> 0.4039
- eval top-3 accuracy: 0.6540
- eval top-5 accuracy: 0.7522

## Playing-Strength Checks

Sampled checkpoint move selection, alternating sides, MCTS128, batch64:

- vs previous checkpoint: 16-0
- vs YaneuraOu MaterialLv1 `go nodes 1`: 0-16

## Integrity

SHA256:

```text
5ef88aee8edf8d79bc569ea939a853f121c16618296402f97447ff65a18c470c
```
