# Shogi Generated Game Throughput

This document records throughput measurements for generated shogi games.

`runs/` is disposable. Measurements that should survive must be summarized here
instead of relying on run-local paths.

## Measurement Conditions

Unless noted otherwise:

- generation entrypoint: `shogi-arena-agent/scripts/generate_shogi_games.py`
- board backend: `cshogi`
- max plies: 320
- batching path: generated-game measurements use multi-position NN leaf
  batching across active games

## Summary

- The checkpoint self-play central evaluator path was measured on 2026-06-02
  with MCTS16 / 16 games on an A40 secure Pod. Increasing worker threads merged
  model requests into larger central batches, but generator CPU stayed near one
  core and GPU utilization remained very low. In this small run, `w2_c4` was the
  fastest at 8.41 plies/sec; `w4_c4` and `w1_c16` were close but slower.
- After specializing the checkpoint self-play tree, the 2026-06-02 L4 secure
  Pod measurement still showed low GPU utilization and generator CPU near one
  core. This is not a strict before/after comparison because the GPU and Pod
  shape changed.
- The 2026-06-03 process-worker measurement on an L4 secure Pod improved
  central batch size as workers increased. `process_w4_c4` was best in that run
  at 8.29 plies/sec, but GPU utilization stayed below 4% on average.
- The 2026-06-04 64-game process-worker measurement showed the same bottleneck
  more clearly. Increasing workers from 4 to 16 improved central batch size and
  plies/sec, but GPU utilization stayed around 2-3%; central model-call wall
  time dominated generation wall time.
- The 2026-06-05 central evaluator phase profile found the main cost inside
  output-side feature construction, not GPU forward. In `w4_c4_s16_b32_g64`,
  output feature build took 1016.99s of 1254.33s backend time, while model
  forward took 83.58s.
- The 2026-06-06 direct USI parser measurement improved the action-plane
  self-play path. On an RTX 4000 Ada secure Pod,
  `w16_c4_s16_b32_g64_direct_usi` reached 64.17 plies/sec. Output feature build
  was no longer the dominant backend phase.
- The 2026-06-06 direct SFEN position parser measurement improved the
  minimal-split-global input path. On an L4 secure Pod,
  `w16_c4_s16_b32_g64_direct_position_l4` reached 101.05 plies/sec, with
  position feature build down to 5.77s of 65.76s backend time.
- Increasing the same direct-position measurement from 64 to 128 games improved
  central batch fill from 62.50% to 71.06% and throughput from 101.05 to
  110.30 plies/sec.
- After batching multiple MCTS leaves within a single self-play position,
  `w16_c1_s16_b32_g128_batched_mcts_l4` filled central batches to 94.91% and
  reached 113.28 plies/sec. This made `c1` viable, but the gain over
  `w16_c4_s16_b32_g128_direct_position_l4` was modest.
- After gathering action-plane legal logits in one pass,
  `w16_c1_s16_b32_g128_fast_output_l4` reached 133.09 plies/sec. Backend
  output feature build fell from 18.23s to 8.77s, and output decode fell from
  16.24s to 12.88s.
- With the same fast output path, increasing only the central evaluator batch
  limit from 32 to 64 reached 150.76 plies/sec.
- BF16 autocast was the strongest measured inference setting. It reached
  222.89 plies/sec and reduced backend model forward time from 45.32s to
  18.12s without changing the checkpoint.
- Combining BF16 with central batch 64 did not improve over BF16 with central
  batch 32: 220.61 plies/sec versus 222.89 plies/sec. BF16 with central batch
  32 remains the better measured setting.
- `torch.compile` did not materially reduce model forward time in this run:
  44.65s versus 45.32s for the fast-output FP32 baseline.
- With MCTS256, full legal expansion, BF16, and the compact self-play MCTS node
  layout, `w16_c1_s256_b64_g16_bf16_compact_node_a4000` reached 13.83
  sec/game on an RTX A4000 community Pod. GPU utilization was still only
  25.95% on average, so generation was not GPU-compute dominated.
- With aligned move priors and array-backed self-play MCTS child stats,
  `w16_c1_s256_b64_g16_bf16_aligned_array_a4000` reduced expand time from
  105.68s to 17.02s and reached 10.87 plies/sec. Wall time per game was higher
  because average game length increased from 122.5 to 199.2 plies.
- Removing the redundant self-play MCTS parent-edge list reduced selection
  time further. `w16_c1_s256_b64_g16_bf16_no_edge_parent_a4000` measured
  expand 13.09s and selection 53.02s with MCTS256 unchanged.
- Earlier MCTS128 self-play measurements were much slower than the older
  light-search MCTS16 measurements. On an L4 secure Pod, 64 games took about
  21.6 minutes, which extrapolates to about 46.1 hours for 8192 games.
- Earlier MCTS128 measurements underfilled NN leaf evaluation batches:
  average actual batch size was about 6.4 with a batch limit of 64.
- Increasing per-process concurrency from 8 to 16 only helps if each worker has
  enough games to keep active. In `w8_c16_s128_b64_g64_a40`, each of 8 workers
  received only 8 games, so the effective per-worker concurrency stayed capped
  at 8.
- In the recorded MCTS128 measurements, larger per-worker batches did not
  improve throughput: `w4_c16_s128_b64_g64_l4` filled batches better than
  `w8_c8_s128_b64_g64_l4`, but had lower plies/sec.
- The valid `w8_c16_s128_b64_g128_a40` measurement did increase actual NN leaf
  eval batch size to 11.49, but throughput was still lower than
  `w8_c8_s128_b64_g64_l4`. This run also had longer games and more max-plies
  draws, so compare by plies/sec, not only wall time.
- On the same A40 / 9 vCPU shape, `w8_c16_s128_b64_g128_a40` was only slightly
  faster than `w8_c8_s128_b64_g128_a40` despite much larger batches. This
  suggests vCPU / Pod shape may matter, but larger per-worker concurrency alone
  is not a strong throughput lever in the current implementation.
- Generation worker processes improved throughput materially in recorded
  light-search self-play measurements.
- On the observed 6 vCPU Pod, worker 8 did not materially improve over worker 6.
- Increasing `NN leaf eval batch limit` from 32 to 64 did not improve
  throughput in the recorded worker 6 comparison.
- Requesting more vCPU materially improved throughput in the recorded worker 8
  comparison.
- Wall time is sensitive to game length. Use `plies/sec` when comparing
  throughput across self-play settings.
- `System RAM used` is not included because it was recorded from
  container-visible `/proc/meminfo`; in these runs it exposed a larger memory
  total than the Pod settings. Use `Generator RSS` for process-scoped memory
  comparison.

## Checkpoint Self-Play Central Evaluator Measurements

These measurements use
`shogi-arena-agent/scripts/generate_checkpoint_self_play_games.py`, where
workers share one central checkpoint evaluator.

| Case | Date | Model | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Runtime image | Total games | Self-play workers | Concurrent games per worker | MCTS simulations per move | NN leaf eval batch limit | Central batch avg | Central batch max | Central batch fill | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | Generator RSS | Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `central_w1_c4_s16_b32_g16_a40` | 2026-06-02 | shogi-minimal-split-global-action-plane | A40 | 9 vCPU, 50 GiB | secure | CA-MTL-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 4 | 16 | 32 | 3.65 | 4 | 11.40% | 174.4 | 409.33 | 6.82 | 3.50% | 9.00% | 393 MiB / 46068 MiB | 93.75% | 156.00% | 946 MiB | Baseline for the central evaluator path. |
| `central_w2_c4_s16_b32_g16_a40` | 2026-06-02 | shogi-minimal-split-global-action-plane | A40 | 9 vCPU, 50 GiB | secure | CA-MTL-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 2 | 4 | 16 | 32 | 6.12 | 8 | 19.13% | 146.9 | 279.54 | 8.41 | 3.46% | 6.00% | 395 MiB / 46068 MiB | 98.05% | 162.00% | 976 MiB | Best plies/sec in this run. Central batches merged across two workers. |
| `central_w4_c4_s16_b32_g16_a40` | 2026-06-02 | shogi-minimal-split-global-action-plane | A40 | 9 vCPU, 50 GiB | secure | CA-MTL-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 4 | 4 | 16 | 32 | 8.43 | 16 | 26.33% | 175.1 | 346.81 | 8.08 | 2.87% | 6.00% | 415 MiB / 46068 MiB | 100.92% | 160.00% | 967 MiB | Larger central batches than `central_w2_c4`, but lower plies/sec. |
| `central_w1_c16_s16_b32_g16_a40` | 2026-06-02 | shogi-minimal-split-global-action-plane | A40 | 9 vCPU, 50 GiB | secure | CA-MTL-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 16 | 16 | 32 | 7.91 | 16 | 24.71% | 159.6 | 318.87 | 8.01 | 3.09% | 8.00% | 415 MiB / 46068 MiB | 98.75% | 144.00% | 942 MiB | Same active-game scale as `central_w4_c4`; similar throughput without multiple worker threads. |
| `central_tree_w1_c4_s16_b32_g16_l4` | 2026-06-02 | shogi-minimal-split-global-action-plane | L4 | 12 vCPU, 201 GiB | secure | EUR-IS-2 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 4 | 16 | 32 | 3.48 | 4 | 10.89% | 171.9 | 356.21 | 7.72 | 4.90% | 20.00% | 302 MiB / 23034 MiB | 94.00% | 186.00% | 964 MiB | Specialized self-play tree; L4 run, not directly comparable to A40 baseline. |
| `central_tree_w2_c4_s16_b32_g16_l4` | 2026-06-02 | shogi-minimal-split-global-action-plane | L4 | 12 vCPU, 201 GiB | secure | EUR-IS-2 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 2 | 4 | 16 | 32 | 6.21 | 8 | 19.40% | 189.1 | 434.38 | 6.96 | 3.86% | 16.00% | 304 MiB / 23034 MiB | 96.71% | 145.00% | 1001 MiB | Specialized self-play tree; L4 run, not directly comparable to A40 baseline. |
| `central_tree_w4_c4_s16_b32_g16_l4` | 2026-06-02 | shogi-minimal-split-global-action-plane | L4 | 12 vCPU, 201 GiB | secure | EUR-IS-2 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 4 | 4 | 16 | 32 | 8.68 | 16 | 27.14% | 177.4 | 366.34 | 7.75 | 3.39% | 11.00% | 516 MiB / 23034 MiB | 100.88% | 187.00% | 990 MiB | Best plies/sec in this L4 run but only slightly above `central_tree_w1_c4`. |
| `central_tree_w1_c16_s16_b32_g16_l4` | 2026-06-02 | shogi-minimal-split-global-action-plane | L4 | 12 vCPU, 201 GiB | secure | EUR-IS-2 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 16 | 16 | 32 | 9.23 | 16 | 28.83% | 185.9 | 400.51 | 7.43 | 3.05% | 15.00% | 324 MiB / 23034 MiB | 99.84% | 187.00% | 959 MiB | Same active-game scale as `central_tree_w4_c4`; still low GPU utilization. |
| `process_w1_c4_s16_b32_g16_l4` | 2026-06-03 | shogi-minimal-split-global-action-plane | L4 | 18 vCPU, 71 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 4 | 16 | 32 | 3.57 | 4 | 11.17% | 169.7 | 460.23 | 5.90 | 2.58% | 7.00% | 302 MiB / 23034 MiB | 97.16% | 270.00% | 924 MiB | Process-worker path; slower than the 2026-06-02 L4 single-worker run. |
| `process_w2_c4_s16_b32_g16_l4` | 2026-06-03 | shogi-minimal-split-global-action-plane | L4 | 18 vCPU, 71 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 2 | 4 | 16 | 32 | 4.24 | 8 | 13.24% | 141.2 | 314.75 | 7.18 | 3.02% | 7.00% | 304 MiB / 23034 MiB | 96.18% | 234.00% | 948 MiB | Process workers improved over `process_w1_c4`, but still low GPU utilization. |
| `process_w4_c4_s16_b32_g16_l4` | 2026-06-03 | shogi-minimal-split-global-action-plane | L4 | 18 vCPU, 71 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 4 | 4 | 16 | 32 | 6.60 | 16 | 20.61% | 177.1 | 342.05 | 8.29 | 2.90% | 7.00% | 324 MiB / 23034 MiB | 100.89% | 235.00% | 941 MiB | Best plies/sec in the 2026-06-03 process-worker run. |
| `process_w1_c16_s16_b32_g16_l4` | 2026-06-03 | shogi-minimal-split-global-action-plane | L4 | 18 vCPU, 71 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 1 | 16 | 16 | 32 | 7.61 | 16 | 23.77% | 154.2 | 329.73 | 7.48 | 2.56% | 8.00% | 324 MiB / 23034 MiB | 103.37% | 269.00% | 939 MiB | Larger within-process batches did not beat `process_w4_c4`. |
| `w4_c4_s16_b32_g64` | 2026-06-04 | shogi-minimal-split-global-action-plane | L4 | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 4 | 4 | 16 | 32 | 9.06 | 16 | 28.30% | 174.4 | 1525.22 | 7.32 | 2.42% | 8.00% | 508 MiB / 23034 MiB | 97.06% | 218.00% | 1054 MiB | 64-game process-worker run. Central model wall 1449.56s of 1525.22s; queue wait avg 0.028s. |
| `w8_c4_s16_b32_g64` | 2026-06-04 | shogi-minimal-split-global-action-plane | L4 | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 8 | 4 | 16 | 32 | 13.63 | 32 | 42.61% | 161.0 | 1332.27 | 7.73 | 2.33% | 12.00% | 556 MiB / 23034 MiB | 100.52% | 240.00% | 1057 MiB | Larger central batches, but only a small throughput gain. Central model wall 1284.14s; queue wait avg 0.053s. |
| `w16_c4_s16_b32_g64` | 2026-06-04 | shogi-minimal-split-global-action-plane | L4 | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 16 | 4 | 16 | 32 | 18.79 | 32 | 58.72% | 158.1 | 1152.04 | 8.79 | 2.60% | 11.00% | 556 MiB / 23034 MiB | 102.21% | 250.00% | 1054 MiB | Best of this 64-game run, but still low GPU utilization. Central model wall 1112.99s; queue wait avg 0.120s. |
| `w4_c4_s16_b32_g64_phase_profile` | 2026-06-05 | shogi-minimal-split-global-action-plane | L4 | 21 vCPU, 83 GiB | secure | EUR-IS-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 4 | 4 | 16 | 32 | 7.95 | 16 | 24.85% | 164.4 | 1347.12 | 7.81 | 2.57% | 8.00% | 516 MiB / 23034 MiB | 98.58% | 258.00% | 1045 MiB | Phase profile. Backend total 1254.33s: output feature build 1016.99s, model forward 83.58s, output decode 84.76s, position feature build 43.58s. |
| `w4_c4_s16_b32_g64_direct_usi` | 2026-06-06 | shogi-minimal-split-global-action-plane | RTX 4000 Ada | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.26/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 4 | 4 | 16 | 32 | 8.93 | 16 | 27.91% | 160.5 | 227.41 | 45.18 | 15.04% | 21.00% | 289 MiB / 20475 MiB | 90.14% | 219.00% | 1066 MiB | Direct USI parser. Backend total 156.33s: output feature build 24.32s, model forward 70.66s, output decode 17.18s, position feature build 40.93s. |
| `w8_c4_s16_b32_g64_direct_usi` | 2026-06-06 | shogi-minimal-split-global-action-plane | RTX 4000 Ada | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.26/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 8 | 4 | 16 | 32 | 14.91 | 32 | 46.58% | 166.6 | 181.72 | 58.69 | 17.23% | 24.00% | 337 MiB / 20475 MiB | 105.88% | 255.00% | 1053 MiB | Direct USI parser. Backend total 136.19s: output feature build 23.95s, model forward 48.84s, output decode 17.76s, position feature build 43.38s. |
| `w16_c4_s16_b32_g64_direct_usi` | 2026-06-06 | shogi-minimal-split-global-action-plane | RTX 4000 Ada | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.26/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 16 | 4 | 16 | 32 | 22.05 | 32 | 68.92% | 188.4 | 187.95 | 64.17 | 18.96% | 27.00% | 337 MiB / 20475 MiB | 108.66% | 219.00% | 1080 MiB | Direct USI parser. Backend total 145.37s: output feature build 28.60s, model forward 45.07s, output decode 20.32s, position feature build 49.48s. |
| `w4_c4_s16_b32_g64_direct_position_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 4 | 4 | 16 | 32 | 10.84 | 16 | 33.88% | 171.2 | 146.51 | 74.78 | 23.33% | 33.00% | 324 MiB / 23034 MiB | 74.61% | 100.00% | 1092 MiB | Direct SFEN position parser. Backend total 84.15s: output feature build 17.15s, model forward 45.29s, output decode 12.19s, position feature build 6.98s. |
| `w8_c4_s16_b32_g64_direct_position_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 8 | 4 | 16 | 32 | 14.88 | 32 | 46.51% | 155.9 | 115.93 | 86.05 | 26.12% | 47.00% | 372 MiB / 23034 MiB | 92.56% | 122.00% | 1084 MiB | Direct SFEN position parser. Backend total 75.68s: output feature build 16.53s, model forward 38.45s, output decode 11.76s, position feature build 6.80s. |
| `w16_c4_s16_b32_g64_direct_position_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 16 | 4 | 16 | 32 | 20.00 | 32 | 62.50% | 149.5 | 94.71 | 101.05 | 30.70% | 47.00% | 376 MiB / 23034 MiB | 104.34% | 112.00% | 1087 MiB | Direct SFEN position parser. Backend total 65.76s: output feature build 14.13s, model forward 34.10s, output decode 10.04s, position feature build 5.77s. |
| `w16_c4_s16_b32_g128_direct_position_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 4 | 16 | 32 | 22.74 | 32 | 71.06% | 160.6 | 186.35 | 110.30 | 32.96% | 50.00% | 372 MiB / 23034 MiB | 105.44% | 114.00% | 1270 MiB | Direct SFEN position parser. Backend total 133.58s: output feature build 28.77s, model forward 69.18s, output decode 20.68s, position feature build 11.85s. |
| `w16_c1_s16_b32_g128_batched_mcts_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 30.37 | 32 | 94.91% | 104.2 | 117.79 | 113.28 | 32.41% | 42.00% | 356 MiB / 23034 MiB | 96.04% | 131.00% | 1091 MiB | Batched self-play MCTS. Backend total 90.17s: output feature build 18.23s, model forward 43.71s, output decode 16.24s, position feature build 9.78s. |
| `w16_c1_s16_b32_g128_fast_output_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 30.53 | 32 | 95.41% | 107.9 | 103.76 | 133.09 | 39.38% | 57.00% | 357 MiB / 23034 MiB | 103.05% | 136.00% | 1062 MiB | Batched self-play MCTS with one-pass action-plane legal-logit gather. Backend total 78.85s: output feature build 8.77s, model forward 45.32s, output decode 12.88s, position feature build 9.81s. |
| `w16_c1_s16_leaf32_central64_g128_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 51.25 | 64 | 80.07% | 99.7 | 84.64 | 150.76 | 46.43% | 65.00% | 424 MiB / 23034 MiB | 100.16% | 104.00% | 1032 MiB | Central evaluator batch limit 64, MCTS leaf batch limit 32. Backend total 64.03s: output feature build 4.82s, model forward 42.59s, output decode 8.42s, position feature build 7.21s. |
| `w16_c1_s16_b32_g128_bf16_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 30.94 | 32 | 96.68% | 95.9 | 55.09 | 222.89 | 25.79% | 35.00% | 336 MiB / 23034 MiB | 101.33% | 106.00% | 1144 MiB | BF16 autocast. Backend total 38.65s: output feature build 4.23s, model forward 18.12s, output decode 8.21s, position feature build 6.89s. |
| `w16_c1_s16_leaf32_central64_g128_bf16_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 53.74 | 64 | 83.97% | 107.9 | 62.59 | 220.61 | 28.53% | 40.00% | 399 MiB / 23034 MiB | 104.86% | 109.00% | 1167 MiB | BF16 autocast with central evaluator batch limit 64 and MCTS leaf batch limit 32. Backend total 43.10s: output feature build 4.69s, model forward 20.89s, output decode 8.89s, position feature build 7.45s. |
| `w16_c1_s16_b32_g128_compile_l4` | 2026-06-06 | shogi-minimal-split-global-action-plane | L4 | 6 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 1 | 16 | 32 | 29.84 | 32 | 93.26% | 109.4 | 95.75 | 146.24 | 41.44% | 59.00% | 354 MiB / 23034 MiB | 100.24% | 119.00% | 1215 MiB | `torch.compile`. Backend total 70.74s: output feature build 5.98s, model forward 44.65s, output decode 10.09s, position feature build 8.44s. |
| `w16_c1_s256_b64_g16_bf16_compact_node_a4000` | 2026-06-09 | shogi-minimal-split-global-position-action-plane-mcts256-full | RTX A4000 | 14 vCPU, 62 GiB | community | SE | $0.17/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 16 | 1 | 256 | 64 | 59.60 | 64 | 93.13% | 122.5 | 221.21 | 8.86 | 25.95% | 38.00% | 367 MiB / 16376 MiB | 106.72% | 145.00% | 1052 MiB | Compact self-play MCTS node layout. Full legal expansion, no top-k pruning. Backend total 141.53s: expand 105.68s, selection 30.13s, model forward 60.73s, output decode 38.00s, output feature build 19.68s, position feature build 20.50s. |
| `w16_c1_s256_b64_g16_bf16_aligned_array_a4000` | 2026-06-09 | shogi-minimal-split-global-position-action-plane-mcts256-full | RTX A4000 | not recorded | community | not recorded | $0.17/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 16 | 1 | 256 | 64 | 63.26 | 64 | 98.85% | 199.2 | 293.19 | 10.87 | 31.83% | 40.00% | 355 MiB / 16376 MiB | 107.67% | 164.00% | 1084 MiB | Aligned move priors plus array-backed self-play MCTS child stats. Full legal expansion, no top-k pruning. Backend total 218.18s: expand 17.02s, selection 63.50s, model forward 94.68s, output decode 54.20s, output feature build 34.28s, position feature build 31.91s. |
| `w16_c1_s256_b64_g16_bf16_no_edge_parent_a4000` | 2026-06-09 | shogi-minimal-split-global-position-action-plane-mcts256-full | RTX A4000 | 14 vCPU, 62 GiB | community | SE | $0.17/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 16 | 1 | 256 | 64 | 60.11 | 64 | 93.92% | 170.0 | 251.69 | 10.81 | 31.32% | 40.00% | 355 MiB / 16376 MiB | 109.72% | 202.00% | 1076 MiB | Removed redundant parent-edge list from self-play MCTS selection/backprop. Full legal expansion, no top-k pruning. Backend total 183.59s: expand 13.09s, selection 53.02s, model forward 82.49s, output decode 44.75s, output feature build 26.33s, position feature build 27.27s. |
| `w16_c1_s256_b64_g16_bf16_action_indices_a4000` | 2026-06-10 | shogi-minimal-split-global-position-action-plane-mcts256-full | RTX A4000 | 6 vCPU, 62 GiB | community | SK | $0.17/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 16 | 16 | 1 | 256 | 64 | 62.32 | 64 | 97.37% | 189.9 | 302.21 | 10.05 | 29.78% | 38.00% | 371 MiB / 16376 MiB | 101.79% | 136.00% | 1098 MiB | cshogi native moves plus precomputed action indices for action-plane inference. Full legal expansion, no top-k pruning. Phase totals: legal moves 162.17s, expand 16.16s, selection 95.91s. Backend totals: model forward 93.03s, output decode 61.98s, output feature build 25.52s, position feature build 48.91s. |

## Detailed Measurements

Case IDs use
`w<workers>_c<concurrent-games>_s<simulations>_b<batch-limit>_g<total-games>[_suffix]`.

| Case | Date | Players | Model | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Runtime image | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Actual NN leaf eval batch avg | Actual NN leaf eval batch max | Actual NN leaf eval batch fill | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | Generator RSS | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |
| `w8_c8_s128_b64_g64_l4` | 2026-05-30 | checkpoint vs checkpoint | shogi-action-plane-policy-output-minimal-split-global | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 8 | 8 | 128 | 64 | 6.41 | 8 | 10.02% | 126.8 | 1296.08 | 6.26 | 36.08% | 67.00% | avg 2105 MiB, max 2556 MiB / 23034 MiB | 7.69% | 11.68% | not recorded | 63 game_over, 1 max_plies. Extrapolates to about 46.1h and $18.0 for 8192 games on this Pod. |
| `w4_c16_s128_b64_g64_l4` | 2026-05-30 | checkpoint vs checkpoint | shogi-action-plane-policy-output-minimal-split-global | L4 | 16 vCPU, 62 GiB | secure | EU-RO-1 | $0.39/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 16 | 4 | 128 | 64 | 11.42 | 16 | 17.84% | 130.0 | 1614.05 | 5.16 | 19.39% | 58.00% | avg 1096 MiB, max 1368 MiB / 23034 MiB | 11.48% | 27.95% | not recorded | 60 game_over, 4 max_plies. Larger batches than `w8_c8_s128_b64_g64_l4`, but lower throughput; extrapolates to about 57.4h and $22.4 for 8192 games on this Pod. |
| `w8_c16_s128_b64_g64_a40` | 2026-05-30 | checkpoint vs checkpoint | shogi-action-plane-policy-output-minimal-split-global | A40 | 9 vCPU, 50 GiB | secure | EU-SE-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 64 | 16 | 8 | 128 | 64 | 6.41 | 8 | 10.02% | 126.8 | 1223.29 | 6.63 | 41.66% | 75.00% | avg 2753 MiB, max 3294 MiB / 46068 MiB | 11.91% | 16.46% | not recorded | 63 game_over, 1 max_plies. `concurrent-games-per-process=16` did not increase actual batch size because 64 games across 8 workers leaves only 8 games per worker. Extrapolates to about 43.5h and $19.1 for 8192 games on this Pod. |
| `w8_c16_s128_b64_g128_a40` | 2026-05-30 | checkpoint vs checkpoint | shogi-action-plane-policy-output-minimal-split-global | A40 | 9 vCPU, 50 GiB | secure | EU-SE-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 16 | 8 | 128 | 64 | 11.49 | 16 | 17.96% | 143.8 | 3204.50 | 5.74 | 28.29% | 77.00% | avg 2539 MiB, max 3731 MiB / 46068 MiB | 16.72% | 28.73% | not recorded | 115 game_over, 13 max_plies. This is the valid `w8_c16` measurement: each worker received 16 games. Larger batches did not improve plies/sec; extrapolates to about 57.0h and $25.1 for 8192 games on this Pod. |
| `w8_c8_s128_b64_g128_a40` | 2026-05-30 | checkpoint vs checkpoint | shogi-action-plane-policy-output-minimal-split-global | A40 | 9 vCPU, 50 GiB | secure | EU-SE-1 | $0.44/hr | runpod-torch-v280 / torch 2.8.0+cu128 | 128 | 8 | 8 | 128 | 64 | 6.73 | 8 | 10.51% | 143.8 | 3295.55 | 5.59 | 31.95% | 82.00% | avg 2402 MiB, max 3278 MiB / 46068 MiB | 12.46% | 23.49% | not recorded | 115 game_over, 13 max_plies. Same game count, seed, GPU, vCPU/RAM, and end-reason profile as `w8_c16_s128_b64_g128_a40`; smaller batches were only slightly slower. Extrapolates to about 58.6h and $25.8 for 8192 games on this Pod. |
| `w6_c8_s16_b32_g48_l4` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | not recorded | 48 | 8 | 6 | 16 | 32 | 6.32 | 8 | 19.75% | 209.6 | 125.92 | 79.88 | 52.96% | 79.00% | 2270 MiB / 23034 MiB | 694.01% | 1195.40% | 5841 MiB | 29 game_over, 19 max_plies. |
| `w6_c8_s16_b64_g48_l4` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | not recorded | 48 | 8 | 6 | 16 | 64 | 6.34 | 8 | 9.91% | 189.9 | 140.40 | 64.91 | 41.48% | 76.00% | 2294 MiB / 23034 MiB | 665.20% | 1176.70% | 5851 MiB | 35 game_over, 13 max_plies. |
| `w6_c16_s16_b32_g96` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | not recorded | 96 | 16 | 6 | 16 | 32 | 12.14 | 16 | 37.93% | 209.4 | 224.96 | 89.36 | 47.13% | 73.00% | 3562 MiB / 23034 MiB | 686.42% | 1190.40% | 6164 MiB | 58 game_over, 38 max_plies. |
| `w6_c16_s16_b64_g96` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | not recorded | 96 | 16 | 6 | 16 | 64 | 11.75 | 16 | 18.35% | 171.7 | 181.91 | 90.61 | 48.65% | 75.00% | 3122 MiB / 23034 MiB | 705.50% | 1190.00% | 6060 MiB | 75 game_over, 21 max_plies. |
| `w1_c8_s128_b64_g8` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX A5000 | 9 vCPU, 50 GiB | secure | EU-SE-1 | $0.27/hr | not recorded | 8 | 8 | 1 | 128 | 64 | 5.09 | 8 | 7.95% | 151.9 | 378.39 | 3.21 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Completed. End reasons: 6 game_over, 2 max_plies. Result: black 5, white 1, draws 2. |
| `w1_c16_s16_b32_g16` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | not recorded | 16 | 16 | 1 | 16 | 32 | not recorded | not recorded | not recorded | 212.1 | 266.82 | 12.72 | 4.37% | 11.00% | 640 MiB / 20475 MiB | 102.42% | 142.00% | 995 MiB | 2026-05-13 current-code profile. Measured phase share: expand 63.81%, selection 23.38%. |
| `w2_c8_s16_b32_g16` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | not recorded | 16 | 8 | 2 | 16 | 32 | not recorded | not recorded | not recorded | 212.0 | 178.58 | 18.99 | 9.95% | 29.00% | 703 MiB / 20475 MiB | 172.32% | 308.30% | 1983 MiB | 2026-05-13 current-code profile. Measured phase share: expand 64.08%, selection 22.75%. |
| `w4_c8_s16_b32_g32` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | not recorded | 32 | 8 | 4 | 16 | 32 | not recorded | not recorded | not recorded | 247.0 | 193.87 | 40.77 | 24.60% | 51.00% | 1415 MiB / 20475 MiB | 356.25% | 498.10% | 3917 MiB | 2026-05-13 current-code profile. Measured phase share: expand 62.76%, selection 23.94%. |
| `w6_c8_s16_b32_g48_rtx4000ada` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | not recorded | 48 | 8 | 6 | 16 | 32 | not recorded | not recorded | not recorded | 241.7 | 230.17 | 50.40 | 33.64% | 59.00% | 2089 MiB / 20475 MiB | 485.50% | 567.50% | 5825 MiB | 2026-05-13 current-code profile. Measured phase share: expand 65.81%, selection 21.63%. |
| `w6_c8_s16_b64_g48_rtx4000ada` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | not recorded | 48 | 8 | 6 | 16 | 64 | not recorded | not recorded | not recorded | 201.1 | 194.30 | 49.68 | 34.17% | 58.00% | 2063 MiB / 20475 MiB | 447.00% | 570.70% | 5678 MiB | 18 of 48 games reached max plies. |
| `w8_c8_s16_b32_g64_6vcpu` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 6 vCPU, 31 GiB | community | `EU-RO-1` requested; assigned US | $0.20/hr | not recorded | 64 | 8 | 8 | 16 | 32 | not recorded | not recorded | not recorded | 227.4 | 279.35 | 52.10 | 35.14% | 53.00% | 2816 MiB / 20475 MiB | 463.94% | 586.80% | 7555 MiB | 29 of 64 games reached max plies. |
| `w8_c8_s16_b32_g64_9vcpu` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 9 vCPU, 50 GiB | secure | `EU-RO-1` | $0.26/hr | not recorded | 64 | 8 | 8 | 16 | 32 | not recorded | not recorded | not recorded | 225.9 | 147.35 | 98.10 | 54.48% | 84.00% | 2777 MiB / 20475 MiB | 643.07% | 830.40% | 7825 MiB | 28 of 64 games reached max plies. |
