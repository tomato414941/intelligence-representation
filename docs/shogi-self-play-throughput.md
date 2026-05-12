# Shogi Self-Play Throughput

This document records facts for planning shogi self-play data generation. It is
not a model-quality report and it is not a complete run history.

## Current Path

Self-play generation currently runs through `shogi-arena-agent`:

```text
scripts/generate_shogi_games.py
```

Online Replay calls that script from this repository and stores generated games
as shogi game-record JSONL before splitting them into train/eval examples.

## Current Bottleneck

The observed bottleneck is self-play generation throughput. Full-length shogi
games can be long under the standard 320-ply cap, so a small number of games can
still take minutes.

The generator now records:

- progress lines every configured number of plies
- generation wall time
- plies per second
- MCTS request, model, non-model, and phase timing summaries

Online Replay can pass `--generation-progress-every-plies` and stores each
cycle's `generation-summary.json`.

## 2026-05-12 Throughput Grid

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| games per case | 4 |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 571.164s |
| remote workload runtime | 514.209s |
| estimated total cost | about $0.03 |

Measured results:

| Case | Concurrent games per process | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `p1_b16` | 1 | 16 | 16 | 110.5 | 55.00 | 8.04 |
| `p2_b16` | 2 | 16 | 16 | 304.5 | 114.64 | 10.62 |
| `p4_b16` | 4 | 16 | 16 | 185.2 | 74.26 | 9.98 |
| `p4_b32` | 4 | 16 | 32 | 251.5 | 82.40 | 12.21 |
| `p4_s32_b32` | 4 | 32 | 32 | 221.2 | 179.84 | 4.92 |

Observed best in this small grid:

```text
concurrent-games-per-process=4
mcts-simulations-per-move=16
nn-leaf-eval-batch-limit=32
```

## 2026-05-12 RTX 4000 Ada GPU Utilization Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| vCPU/RAM | 6 vCPU, 31 GiB RAM |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 560.415s |
| remote workload runtime | 505.343s |
| estimated total cost | about $0.03 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `p4_s16_b32` | 4 | 4 | 1 | 16 | 32 | 176.5 | 64.57 | 10.93 | 5.78% | 10.00% | 322 MiB / 20475 MiB | 106.28% | 140.00% | not recorded | not recorded | 1 of 4 games reached max plies. |
| `p8_s16_b32` | 8 | 8 | 1 | 16 | 32 | 295.9 | 159.16 | 14.87 | 5.73% | 10.00% | 354 MiB / 20475 MiB | 103.23% | 137.00% | not recorded | not recorded | 6 of 8 games reached max plies. |
| `p16_s16_b32` | 16 | 16 | 1 | 16 | 32 | 255.2 | 271.63 | 15.04 | 5.05% | 9.00% | 422 MiB / 20475 MiB | 101.73% | 138.00% | not recorded | not recorded | 10 of 16 games reached max plies. |

## 2026-05-12 RTX 4000 Ada Generation Worker Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| vCPU/RAM | 6 vCPU, 31 GiB RAM |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 667.394s |
| remote workload runtime | 623.491s |
| estimated total cost | about $0.04 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `p4_s16_b32` | 4 | 4 | 1 | 16 | 32 | 177.8 | 62.64 | 11.35 | 6.22% | 10.00% | 322 MiB / 20475 MiB | 106.94% | 139.00% | 39907 MiB / 257818 MiB | 1042 MiB | 1 of 4 games reached max plies. |
| `p8_s16_b32` | 8 | 8 | 1 | 16 | 32 | 199.1 | 125.53 | 12.69 | 5.45% | 10.00% | 368 MiB / 20475 MiB | 103.94% | 139.00% | 40970 MiB / 257818 MiB | 951 MiB | 3 of 8 games reached max plies. |
| `p16_s16_b32` | 16 | 16 | 1 | 16 | 32 | 217.9 | 238.56 | 14.62 | 5.29% | 10.00% | 474 MiB / 20475 MiB | 101.71% | 125.00% | 40931 MiB / 257818 MiB | 965 MiB | 7 of 16 games reached max plies. |
| `w4_c8_s16_b32` | 32 | 8 | 4 | 16 | 32 | 238.3 | 182.34 | 41.82 | 24.38% | 55.00% | 1415 MiB / 20475 MiB | 366.41% | 469.20% | 42863 MiB / 257818 MiB | 3838 MiB | 13 of 32 games reached max plies. |

Observed facts:

- `w4_c8_s16_b32` reached 41.82 plies/sec, about 2.9x the same-run
  `p16_s16_b32` throughput.
- Generator CPU average rose from about 1 core to about 3.7 cores.
- GPU utilization average rose from about 5% to about 24%.
- Generator RSS rose to about 3.8 GiB.
- `System RAM used` is recorded from container-visible `/proc/meminfo`; in this
  run it exposed a larger memory total than the Pod's 31 GiB setting. Use
  `Generator RSS` for process-scoped memory comparison.

## 2026-05-12 RTX 4000 Ada Generation Worker Scaling Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| data center | not pinned; assigned machine location was US |
| vCPU/RAM | 6 vCPU, 31 GiB RAM |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 906.726s |
| remote workload runtime | 841.622s |
| estimated total cost | about $0.05 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `w1_c16_s16_b32` | 16 | 16 | 1 | 16 | 32 | 195.5 | 257.17 | 12.16 | 4.44% | 10.00% | 542 MiB / 20475 MiB | 101.93% | 140.00% | 42275 MiB / 257818 MiB | 968 MiB | 5 of 16 games reached max plies. |
| `w2_c8_s16_b32` | 16 | 8 | 2 | 16 | 32 | 257.4 | 187.10 | 22.01 | 10.60% | 30.00% | 705 MiB / 20475 MiB | 205.55% | 300.80% | 41610 MiB / 257818 MiB | 1946 MiB | 9 of 16 games reached max plies. |
| `w4_c8_s16_b32` | 32 | 8 | 4 | 16 | 32 | 215.4 | 178.59 | 38.59 | 24.95% | 58.00% | 1415 MiB / 20475 MiB | 381.92% | 498.30% | 42612 MiB / 257818 MiB | 3793 MiB | 12 of 32 games reached max plies. |
| `w6_c8_s16_b32` | 48 | 8 | 6 | 16 | 32 | 230.8 | 208.04 | 53.24 | 35.65% | 62.00% | 2113 MiB / 20475 MiB | 478.39% | 569.20% | 44580 MiB / 257818 MiB | 5667 MiB | 22 of 48 games reached max plies. |

Observed facts:

- Throughput increased from 12.16 to 53.24 plies/sec between 1 and 6 generation
  worker processes.
- GPU utilization average increased from 4.44% to 35.65%.
- Generator CPU average increased from about 1.0 core to about 4.8 cores on a
  6 vCPU Pod.
- Generator RSS increased from about 1.0 GiB to about 5.5 GiB.
- The 6-worker case was still below full GPU saturation.

## 2026-05-12 RTX 4000 Ada NN Batch Limit Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| data center | not pinned; assigned machine location was US |
| vCPU/RAM | 6 vCPU, 31 GiB RAM |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 264.221s |
| remote workload runtime | 199.024s |
| estimated total cost | about $0.02 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `w6_c8_s16_b64` | 48 | 8 | 6 | 16 | 64 | 201.1 | 194.30 | 49.68 | 34.17% | 58.00% | 2063 MiB / 20475 MiB | 447.00% | 570.70% | 43336 MiB / 257818 MiB | 5678 MiB | 18 of 48 games reached max plies. |

Observed facts:

- Increasing `NN leaf eval batch limit` from 32 to 64 at 6 worker processes did
  not increase GPU utilization in this run.
- The comparable 6-worker batch-32 run recorded 53.24 plies/sec and 35.65% GPU
  utilization average.
- The batch-64 run recorded 49.68 plies/sec and 34.17% GPU utilization average.

## 2026-05-12 RTX 4000 Ada Worker 8 Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| data center | `EU-RO-1` requested; assigned machine location was US |
| vCPU/RAM | 6 vCPU, 31 GiB RAM |
| RunPod rate | $0.20/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 339.589s |
| remote workload runtime | 284.899s |
| estimated total cost | about $0.02 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `w8_c8_s16_b32` | 64 | 8 | 8 | 16 | 32 | 227.4 | 279.35 | 52.10 | 35.14% | 53.00% | 2816 MiB / 20475 MiB | 463.94% | 586.80% | 43959 MiB / 257818 MiB | 7555 MiB | 29 of 64 games reached max plies. |

Observed facts:

- Increasing generation worker processes from 6 to 8 did not increase
  throughput in this run.
- The comparable 6-worker batch-32 run recorded 53.24 plies/sec and 35.65% GPU
  utilization average.
- The 8-worker batch-32 run recorded 52.10 plies/sec and 35.14% GPU utilization
  average.
- Generator RSS increased from about 5.5 GiB at 6 workers to about 7.4 GiB at 8
  workers.

## 2026-05-12 RTX 4000 Ada Worker 8 Secure 9 vCPU Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| cloud type | secure |
| data center | `EU-RO-1` |
| vCPU/RAM | 9 vCPU, 50 GiB RAM |
| RunPod rate | $0.26/hr observed at run time |
| torch/CUDA | RunPod PyTorch 2.8 template, torch 2.8.0+cu128, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | 225.423s |
| remote workload runtime | 152.079s |
| estimated total cost | about $0.02 |

Measured results:

| Case | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | System RAM used | Generator RSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `w8_c8_s16_b32` | 64 | 8 | 8 | 16 | 32 | 225.9 | 147.35 | 98.10 | 54.48% | 84.00% | 2777 MiB / 20475 MiB | 643.07% | 830.40% | 52470 MiB / 257587 MiB | 7825 MiB | 28 of 64 games reached max plies. |

Observed facts:

- Moving from the 6 vCPU community Pod to the 9 vCPU secure Pod increased the
  same `w8_c8_s16_b32` throughput from 52.10 to 98.10 plies/sec.
- GPU utilization average increased from 35.14% to 54.48%.
- Generator CPU average increased from about 4.6 cores to about 6.4 cores.
- Generator RSS stayed in the same range: about 7.4 GiB to about 7.6 GiB.
- The higher vCPU Pod was more expensive per hour, but the measured workload
  completed faster.

## Notes

- Wall time is sensitive to game length. Use `plies/sec` when comparing
  throughput across self-play settings.
- `runs/` is disposable. Measurements that should survive must be summarized in
  this document instead of relying on run-local paths.
- The measured grid used RTX 4000 Ada. Do not assume the same ranking holds on
  RTX 4090 or RTX 5090 without measuring.
- The throughput grid did not record GPU utilization over time. It confirms CUDA
  execution, but not whether the GPU was saturated.
- Increasing simulations from 16 to 32 reduced throughput in this grid.
- Process-level game parallelism is exposed as generation worker processes.
  `concurrent-games-per-process` still batches multiple active games inside one Python
  process.
- Current measurements do not use in-tree leaf selection parallelism; batching
  comes from multiple active games, not multiple pending leaves from one MCTS
  tree.
- The RTX 4000 Ada utilization check recorded low GPU utilization. The generator
  process stayed near one CPU core while GPU memory use remained below 500 MiB.
- The generation worker check recorded higher throughput and higher GPU
  utilization by running multiple generator processes.
- The generation worker scaling check recorded continued throughput gains
  through 6 worker processes on a 6 vCPU RTX 4000 Ada Pod.
- The 6-worker batch limit check did not show an advantage for increasing
  `NN leaf eval batch limit` from 32 to 64.
- The 8-worker check did not improve throughput over 6 workers on the observed
  6 vCPU RTX 4000 Ada Pod.
- The secure 9 vCPU worker-8 check recorded much higher throughput and GPU
  utilization than the 6 vCPU worker-8 check.
