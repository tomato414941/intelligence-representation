# Shogi Learning Experiments

This document records comparison-level facts for shogi learning experiments.

It does not treat `runs/` as a durable source of truth.

## Experiments

| Experiment | Date | Question | Setup | Data | Result | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `shogi-learning-20260515-001` | 2026-05-15 | Does online experience replay improve the checkpoint? | 4 cycles; next-cycle checkpoint = cycle final; start checkpoint ID not recorded; start path `models/d256-h1024-heads8-l6-shogi/checkpoint.pt`; self-play 64/cycle + checkpoint-vs-YaneuraOu 64/cycle with checkpoint black fixed | seeded replay; 512 generated games; 65,992 generated train examples | fixed eval worsened from 2.5003 to 10.9028; no head-to-head match | strength improvement undetermined |
| `shogi-learning-20260516-001` | 2026-05-16 | Can full Qhapaq tensor-cache training run on RunPod, and where is time spent? | RunPod RTX 4090; PyTorch 2.8.0+cu128; 500 steps; batch 512; num_workers 2; policy-only | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 16,384 examples | train loss 4.2630 -> 3.0049; eval loss 4.2699 -> 3.0568; eval accuracy 0.0240 -> 0.2794; 500 training steps took 95.0s; job wall time was 278.8s including 108.0s repo/cache sync | tensor-cache training works after cache identity was made path-relocatable; within the measured training loop, data wait was 24.9s, forward/backward was 3.2s, and optimizer was 0.5s |
| `shogi-learning-20260516-002` | 2026-05-16 | Which DataLoader worker count is fastest for full Qhapaq tensor-cache training? | RunPod A40; PyTorch 2.8.0+cu128; 500 steps per case; batch 512; policy-only | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 16,384 examples | See DataLoader profile below | `num_workers=8` was fastest among 0/2/4/8 on this A40 pod; data wait fell sharply as workers increased |

## DataLoader Profiles

### `shogi-learning-20260516-002`

RunPod job context: A40, 9 vCPU, 50 GB RAM, secure cloud, `EU-SE-1`, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, container disk 80 GB, no network volume. Job wall time was 944.4s, including 101.8s repo/cache sync, 14.9s setup, 768.4s remote training/profile, and 18.0s output sync.

| num_workers | 500-step elapsed sec | steps/sec | data wait sec | forward/backward sec | optimizer sec | eval loss | eval accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 190.2 | 2.63 | 57.9 | 15.1 | 0.8 | 3.0569 | 0.2794 |
| 2 | 169.3 | 2.95 | 33.6 | 14.0 | 0.8 | 3.0569 | 0.2793 |
| 4 | 155.2 | 3.22 | 24.3 | 14.5 | 1.0 | 3.0569 | 0.2794 |
| 8 | 148.5 | 3.37 | 3.1 | 14.5 | 0.9 | 3.0569 | 0.2794 |

Checkpoint paths are run-time paths, not immutable checkpoint identities.

## Follow-Up Issues

- `issues/shogi-checkpoint-immutable-identity.md`
- `issues/shogi-checkpoint-match-evaluation.md`
