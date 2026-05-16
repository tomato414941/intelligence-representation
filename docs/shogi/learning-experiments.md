# Shogi Learning Experiments

This document records comparison-level facts for shogi learning experiments.

It does not treat `runs/` as a durable source of truth.

## Experiments

| Experiment | Date | Question | Setup | Data | Result | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `shogi-learning-20260515-001` | 2026-05-15 | Does online experience replay improve the checkpoint? | 4 cycles; next-cycle checkpoint = cycle final; start checkpoint ID not recorded; start path `models/d256-h1024-heads8-l6-shogi/checkpoint.pt`; self-play 64/cycle + checkpoint-vs-YaneuraOu 64/cycle with checkpoint black fixed | seeded replay; 512 generated games; 65,992 generated train examples | fixed eval worsened from 2.5003 to 10.9028; no head-to-head match | strength improvement undetermined |
| `shogi-learning-20260516-001` | 2026-05-16 | Can full Qhapaq tensor-cache training run on RunPod, and where is time spent? | RunPod RTX 4090; PyTorch 2.8.0+cu128; 500 steps; batch 512; num_workers 2; policy-only | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 16,384 examples | train loss 4.2630 -> 3.0049; eval loss 4.2699 -> 3.0568; eval accuracy 0.0240 -> 0.2794; 500 training steps took 95.0s; job wall time was 278.8s including 108.0s repo/cache sync | tensor-cache training works after cache identity was made path-relocatable; within the measured training loop, data wait was 24.9s, forward/backward was 3.2s, and optimizer was 0.5s |
| `shogi-learning-20260516-002` | 2026-05-16 | Which DataLoader worker count is fastest for full Qhapaq tensor-cache training? | RunPod A40; PyTorch 2.8.0+cu128; 500 steps per case; batch 512; policy-only | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 16,384 examples | See DataLoader profile below | `num_workers=8` was fastest among 0/2/4/8 on this A40 pod; data wait fell sharply as workers increased |
| `shogi-learning-20260516-003` | 2026-05-16 | What does one epoch of full Qhapaq tensor-cache training achieve? | RunPod RTX 4090; PyTorch 2.8.0+cu128; 9,670 steps; batch 512; num_workers 8; policy-only; eval every 1,000 steps; keep last 3 step checkpoints | Qhapaq full tensor cache; 4,951,012 train examples; 262,133 eval examples; eval capped at 65,536 examples | train loss 4.2407 -> 2.0753; eval loss 4.2447 -> 2.1020; eval accuracy 0.0259 -> 0.4039; eval top-3 accuracy 0.6540; eval top-5 accuracy 0.7522; actual steps 9,670; job wall time 1,915.9s including 81.6s repo/cache sync and 1,793.0s remote training; post-training player match beat the starting checkpoint 8-0 with alternating sides, MCTS128, batch64, A40, no draws, no illegal moves, average 85.5 plies | One full Qhapaq epoch trains cleanly, substantially improves held-out move-choice accuracy, and beat the starting checkpoint in a deterministic small match |
| `shogi-learning-20260516-004` | 2026-05-16 | Does sampled move selection change the post-training match picture? | RunPod RTX A5000; PyTorch 2.8.0+cu128; player A = `shogi-learning-20260516-003` final checkpoint; player B = starting checkpoint or YaneuraOu MaterialLv1 `go nodes 1`; 16 games per match; alternating sides; MCTS128; batch64; `self-play` move selection profile for checkpoint players | Same checkpoints as `shogi-learning-20260516-003`; no new training data | Against the starting checkpoint, player A won 16-0 with no draws or illegal moves; all 16 games had unique move sequences; average 90.0 plies. Against YaneuraOu MaterialLv1 nodes1, player A lost 0-16 with no draws or illegal moves; all 16 games had unique move sequences; average 66.125 plies | Sampling removed the repeated-game artifact from the small match. The trained checkpoint beat the starting checkpoint and lost to YaneuraOu MaterialLv1 nodes1 under these settings |
| `shogi-learning-20260516-005` | 2026-05-16 | How much MCTS is needed for the promoted checkpoint to take games from YaneuraOu MaterialLv1 nodes1? | RunPod RTX 4000 Ada Generation; PyTorch 2.8.0+cu128; promoted checkpoint as player A; YaneuraOu MaterialLv1 `go nodes 1` as player B; 4 games per MCTS case; alternating sides; evaluation move selection profile; NN leaf eval batch limit 64; max plies 320 | Same promoted checkpoint as `shogi-learning-20260516-003`; no new training data | See YaneuraOu MCTS sweep below | In this small sweep, MCTS128 already took games and MCTS256 won 3-1. Larger MCTS counts did not improve monotonically in the 4-game sample. Every recorded request stayed below 10 seconds on RTX 4000 Ada Generation |

## YaneuraOu MCTS Sweep

### `shogi-learning-20260516-005`

Player A is the promoted project checkpoint. Player B is YaneuraOu MaterialLv1
with `go nodes 1`. Each case used 4 games with alternating sides.

| MCTS simulations per move | Player A result | Avg plies | Request avg sec | Request p95 sec | Request max sec | NN leaf eval batch fill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2-2 | 134.5 | 0.202 | 0.343 | 0.399 | 0.669 |
| 256 | 3-1 | 90.25 | 0.423 | 0.703 | 0.817 | 0.723 |
| 512 | 1-3 | 73.25 | 0.672 | 1.064 | 1.488 | 0.814 |
| 1024 | 0-4 | 80.0 | 1.479 | 2.584 | 4.479 | 0.757 |
| 2048 | 0-4 | 92.5 | 3.023 | 6.396 | 8.621 | 0.787 |

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

The `shogi-learning-20260516-003` player match used deterministic MCTS settings,
so the 8 games contained two unique move sequences repeated four times each.

The `shogi-learning-20260516-004` player matches used sampled checkpoint move
selection, so every game in each 16-game match had a unique move sequence.

## Follow-Up Issues

- `issues/shogi-checkpoint-immutable-identity.md`
- `issues/shogi-checkpoint-match-evaluation.md`
