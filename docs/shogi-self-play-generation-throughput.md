# Shogi Self-Play Generation Throughput

This document records facts for planning shogi self-play data generation. It is
not a model-quality report and it is not a complete run history.

`runs/` is disposable. Measurements that should survive must be summarized in
this document instead of relying on run-local paths.

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

The generator records:

- progress lines every configured number of plies
- generation wall time
- plies per second
- MCTS request, model, non-model, and phase timing summaries

Online Replay can pass `--generation-progress-every-plies` and stores each
cycle's `generation-summary.json`.

## Measurement Conditions

Unless noted otherwise:

- GPU: not fixed. Record the actual GPU used by each measurement. Current
  candidates are RTX A5000, RTX 4090, and RTX 4000 Ada Generation when
  available.
- template: RunPod PyTorch 2.8, torch 2.8.0+cu128 where recorded
- model: `d256-h1024-heads8-l6-shogi`
- board backend: `cshogi`
- max plies: 320
- player profile: checkpoint self-play MCTS on both sides
- MCTS simulations per move: 16

## Findings

### Concurrent Games

The initial small grid used 4 games per case and did not record GPU utilization.
Within that grid, `concurrent-games-per-process=4`,
`mcts-simulations-per-move=16`, and `nn-leaf-eval-batch-limit=32` had the
highest recorded throughput at 12.21 plies/sec.

### Generation Worker Processes

Single-process generation left the GPU mostly idle. With 6 vCPU / 31 GiB
community Pods, generation worker processes increased throughput through 6
workers:

```text
worker=1: 12.72 plies/sec, GPU avg  4.37%
worker=2: 18.99 plies/sec, GPU avg  9.95%
worker=4: 40.77 plies/sec, GPU avg 24.60%
worker=6: 50.40 plies/sec, GPU avg 33.64%
worker=8: 52.10 plies/sec, GPU avg 35.14%
```

On the observed 6 vCPU Pod, worker 8 did not improve over worker 6.

In the 2026-05-13 current-code worker-scaling profile, the measured non-model
MCTS phases were still dominated by expansion and selection:

```text
worker=1: expand 63.81%, selection 23.38%, legal_moves 8.38%, board_copy 3.17%
worker=2: expand 64.08%, selection 22.75%, legal_moves 8.45%, board_copy 3.53%
worker=4: expand 62.76%, selection 23.94%, legal_moves 8.31%, board_copy 3.64%
worker=6: expand 65.81%, selection 21.63%, legal_moves 7.73%, board_copy 3.52%
```

### NN Batch Limit

Increasing `NN leaf eval batch limit` from 32 to 64 at 6 worker processes did
not improve throughput in the recorded run:

```text
worker=6, batch=32: 53.24 plies/sec, GPU avg 35.65%
worker=6, batch=64: 49.68 plies/sec, GPU avg 34.17%
```

### Pod CPU Allocation

Requesting more vCPU changed the result materially. The 9 vCPU secure Pod
recorded much higher throughput than the 6 vCPU community Pod for the same
worker 8 / batch 32 setting:

```text
6 vCPU community: 52.10 plies/sec, GPU avg 35.14%, CPU avg 463.94%
9 vCPU secure:    98.10 plies/sec, GPU avg 54.48%, CPU avg 643.07%
```

## Detailed Measurements

| Case | Pod vCPU/RAM | Cloud | Data center | Rate | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | Generator RSS | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `w1_c16_s16_b32` | 6 vCPU, 31 GiB | community | US | $0.20/hr | 16 | 16 | 1 | 16 | 32 | 212.1 | 266.82 | 12.72 | 4.37% | 11.00% | 640 MiB / 20475 MiB | 102.42% | 142.00% | 995 MiB | 2026-05-13 current-code profile. Measured phase share: expand 63.81%, selection 23.38%. |
| `w2_c8_s16_b32` | 6 vCPU, 31 GiB | community | US | $0.20/hr | 16 | 8 | 2 | 16 | 32 | 212.0 | 178.58 | 18.99 | 9.95% | 29.00% | 703 MiB / 20475 MiB | 172.32% | 308.30% | 1983 MiB | 2026-05-13 current-code profile. Measured phase share: expand 64.08%, selection 22.75%. |
| `w4_c8_s16_b32` | 6 vCPU, 31 GiB | community | US | $0.20/hr | 32 | 8 | 4 | 16 | 32 | 247.0 | 193.87 | 40.77 | 24.60% | 51.00% | 1415 MiB / 20475 MiB | 356.25% | 498.10% | 3917 MiB | 2026-05-13 current-code profile. Measured phase share: expand 62.76%, selection 23.94%. |
| `w6_c8_s16_b32` | 6 vCPU, 31 GiB | community | US | $0.20/hr | 48 | 8 | 6 | 16 | 32 | 241.7 | 230.17 | 50.40 | 33.64% | 59.00% | 2089 MiB / 20475 MiB | 485.50% | 567.50% | 5825 MiB | 2026-05-13 current-code profile. Measured phase share: expand 65.81%, selection 21.63%. |
| `p1_b16` | not recorded | community | not recorded | $0.20/hr | 4 | 1 | 1 | 16 | 16 | 110.5 | 55.00 | 8.04 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Initial small grid. |
| `p2_b16` | not recorded | community | not recorded | $0.20/hr | 4 | 2 | 1 | 16 | 16 | 304.5 | 114.64 | 10.62 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Initial small grid. |
| `p4_b16` | not recorded | community | not recorded | $0.20/hr | 4 | 4 | 1 | 16 | 16 | 185.2 | 74.26 | 9.98 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Initial small grid. |
| `p4_b32` | not recorded | community | not recorded | $0.20/hr | 4 | 4 | 1 | 16 | 32 | 251.5 | 82.40 | 12.21 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Initial small grid. |
| `p4_s32_b32` | not recorded | community | not recorded | $0.20/hr | 4 | 4 | 1 | 32 | 32 | 221.2 | 179.84 | 4.92 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Initial small grid. |
| `p4_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 4 | 4 | 1 | 16 | 32 | 176.5 | 64.57 | 10.93 | 5.78% | 10.00% | 322 MiB / 20475 MiB | 106.28% | 140.00% | not recorded | 1 of 4 games reached max plies. |
| `p8_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 8 | 8 | 1 | 16 | 32 | 295.9 | 159.16 | 14.87 | 5.73% | 10.00% | 354 MiB / 20475 MiB | 103.23% | 137.00% | not recorded | 6 of 8 games reached max plies. |
| `p16_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 16 | 16 | 1 | 16 | 32 | 255.2 | 271.63 | 15.04 | 5.05% | 9.00% | 422 MiB / 20475 MiB | 101.73% | 138.00% | not recorded | 10 of 16 games reached max plies. |
| `p4_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 4 | 4 | 1 | 16 | 32 | 177.8 | 62.64 | 11.35 | 6.22% | 10.00% | 322 MiB / 20475 MiB | 106.94% | 139.00% | 1042 MiB | 1 of 4 games reached max plies. |
| `p8_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 8 | 8 | 1 | 16 | 32 | 199.1 | 125.53 | 12.69 | 5.45% | 10.00% | 368 MiB / 20475 MiB | 103.94% | 139.00% | 951 MiB | 3 of 8 games reached max plies. |
| `p16_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 16 | 16 | 1 | 16 | 32 | 217.9 | 238.56 | 14.62 | 5.29% | 10.00% | 474 MiB / 20475 MiB | 101.71% | 125.00% | 965 MiB | 7 of 16 games reached max plies. |
| `w4_c8_s16_b32` | 6 vCPU, 31 GiB | community | not recorded | $0.20/hr | 32 | 8 | 4 | 16 | 32 | 238.3 | 182.34 | 41.82 | 24.38% | 55.00% | 1415 MiB / 20475 MiB | 366.41% | 469.20% | 3838 MiB | 13 of 32 games reached max plies. |
| `w1_c16_s16_b32` | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 16 | 16 | 1 | 16 | 32 | 195.5 | 257.17 | 12.16 | 4.44% | 10.00% | 542 MiB / 20475 MiB | 101.93% | 140.00% | 968 MiB | 5 of 16 games reached max plies. |
| `w2_c8_s16_b32` | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 16 | 8 | 2 | 16 | 32 | 257.4 | 187.10 | 22.01 | 10.60% | 30.00% | 705 MiB / 20475 MiB | 205.55% | 300.80% | 1946 MiB | 9 of 16 games reached max plies. |
| `w4_c8_s16_b32` | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 32 | 8 | 4 | 16 | 32 | 215.4 | 178.59 | 38.59 | 24.95% | 58.00% | 1415 MiB / 20475 MiB | 381.92% | 498.30% | 3793 MiB | 12 of 32 games reached max plies. |
| `w6_c8_s16_b32` | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 48 | 8 | 6 | 16 | 32 | 230.8 | 208.04 | 53.24 | 35.65% | 62.00% | 2113 MiB / 20475 MiB | 478.39% | 569.20% | 5667 MiB | 22 of 48 games reached max plies. |
| `w6_c8_s16_b64` | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 48 | 8 | 6 | 16 | 64 | 201.1 | 194.30 | 49.68 | 34.17% | 58.00% | 2063 MiB / 20475 MiB | 447.00% | 570.70% | 5678 MiB | 18 of 48 games reached max plies. |
| `w8_c8_s16_b32` | 6 vCPU, 31 GiB | community | `EU-RO-1` requested; assigned US | $0.20/hr | 64 | 8 | 8 | 16 | 32 | 227.4 | 279.35 | 52.10 | 35.14% | 53.00% | 2816 MiB / 20475 MiB | 463.94% | 586.80% | 7555 MiB | 29 of 64 games reached max plies. |
| `w8_c8_s16_b32` | 9 vCPU, 50 GiB | secure | `EU-RO-1` | $0.26/hr | 64 | 8 | 8 | 16 | 32 | 225.9 | 147.35 | 98.10 | 54.48% | 84.00% | 2777 MiB / 20475 MiB | 643.07% | 830.40% | 7825 MiB | 28 of 64 games reached max plies. |

## Job-Level Context

| Measurement | Total job runtime | Remote workload runtime | Estimated cost |
| --- | ---: | ---: | ---: |
| Initial small grid | 571.164s | 514.209s | about $0.03 |
| GPU utilization check | 560.415s | 505.343s | about $0.03 |
| Generation worker check | 667.394s | 623.491s | about $0.04 |
| Generation worker scaling check | 906.726s | 841.622s | about $0.05 |
| NN batch limit check | 264.221s | 199.024s | about $0.02 |
| Worker 8 check | 339.589s | 284.899s | about $0.02 |
| Worker 8 secure 9 vCPU check | 225.423s | 152.079s | about $0.02 |
| 2026-05-13 current-code worker-scaling profile | 926.557s | 880.379s | about $0.05 |

## Notes

- Wall time is sensitive to game length. Use `plies/sec` when comparing
  throughput across self-play settings.
- Most recorded rows so far used RTX 4000 Ada. Treat GPU type as a measurement
  condition, not as a document-wide default.
- On 2026-05-12, RTX 4000 Ada secure Pod creation with
  `minVCPUPerGPU=10` and `minVCPUPerGPU=12` returned no available instances.
- The initial throughput grid did not record GPU utilization over time. It
  confirms CUDA execution, but not whether the GPU was saturated.
- Increasing simulations from 16 to 32 reduced throughput in the initial grid.
- Process-level game parallelism is exposed as generation worker processes.
  `concurrent-games-per-process` still batches multiple active games inside one
  Python process.
- Current measurements do not use in-tree leaf selection parallelism; batching
  comes from multiple active games, not multiple pending leaves from one MCTS
  tree.
- `System RAM used` was removed from the detailed table because it was recorded
  from container-visible `/proc/meminfo`; in these runs it exposed a larger
  memory total than the Pod settings. Use `Generator RSS` for process-scoped
  memory comparison.
