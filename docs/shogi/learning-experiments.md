# Shogi Learning Experiments

This document is the index for shogi learning experiments. It records the
comparison-level facts only. Per-experiment detail belongs in
`docs/shogi/learning-experiments/`.

It does not treat `runs/` as a durable source of truth.

## Experiments

| Experiment | Date | Method | Starting checkpoint ID | Starting checkpoint path | Experience plan | Cycles | Strength conclusion | Details |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `shogi-learning-20260515-001` | 2026-05-15 | online replay | not recorded | `models/d256-h1024-heads8-l6-shogi/checkpoint.pt` | self-play:64 + checkpoint-vs-USI:64 per cycle | 4 | undetermined; fixed eval worsened | [details](learning-experiments/shogi-learning-20260515-001.md) |

Checkpoint paths are run-time paths, not immutable checkpoint identities.

## Follow-Up Issues

- `issues/shogi-checkpoint-immutable-identity.md`
- `issues/shogi-online-replay-strength-evaluation.md`
- `issues/shogi-online-replay-step-budget-policy.md`
