# Shogi Learning Experiments

This document records comparison-level facts for shogi learning experiments.

It does not treat `runs/` as a durable source of truth.

## Experiments

| Experiment | Date | Question | Setup | Data | Result | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `shogi-learning-20260515-001` | 2026-05-15 | Does online experience replay improve the checkpoint? | 4 cycles; next-cycle checkpoint = cycle final; start checkpoint ID not recorded; start path `models/d256-h1024-heads8-l6-shogi/checkpoint.pt`; self-play 64/cycle + checkpoint-vs-YaneuraOu 64/cycle with checkpoint black fixed | seeded replay; 512 generated games; 65,992 generated train examples | fixed eval worsened from 2.5003 to 10.9028; no head-to-head match | strength improvement undetermined |
| `shogi-learning-20260516-001` | 2026-05-16 | Can full Qhapaq tensor-cache training run on RunPod, and where is time spent? | RunPod RTX 4090; PyTorch 2.8.0+cu128; 500 steps; batch 512; num_workers 2; policy-only | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 16,384 examples | train loss 4.2630 -> 3.0049; eval loss 4.2699 -> 3.0568; eval accuracy 0.0240 -> 0.2794; 500 training steps took 95.0s; job wall time was 278.8s including 108.0s repo/cache sync | tensor-cache training works after cache identity was made path-relocatable; within the measured training loop, data wait was 24.9s, forward/backward was 3.2s, and optimizer was 0.5s |

Checkpoint paths are run-time paths, not immutable checkpoint identities.

## Follow-Up Issues

- `issues/shogi-checkpoint-immutable-identity.md`
- `issues/shogi-checkpoint-match-evaluation.md`
