# Shogi Learning Experiments

This document records comparison-level facts for shogi learning experiments.

It does not treat `runs/` as a durable source of truth.

## Experiments

| Experiment | Date | Question | Setup | Data | Result | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `shogi-learning-20260515-001` | 2026-05-15 | Does online experience replay improve the checkpoint? | 4 cycles; start checkpoint ID not recorded; start path `models/d256-h1024-heads8-l6-shogi/checkpoint.pt`; self-play 64/cycle + checkpoint-vs-USI 64/cycle | seeded replay; 512 generated games; 65,992 generated train examples | fixed eval worsened from 2.5003 to 10.9028; no head-to-head match | strength improvement undetermined |

Checkpoint paths are run-time paths, not immutable checkpoint identities.

## Follow-Up Issues

- `issues/shogi-checkpoint-immutable-identity.md`
- `issues/shogi-online-replay-strength-evaluation.md`
- `issues/shogi-online-replay-step-budget-policy.md`
