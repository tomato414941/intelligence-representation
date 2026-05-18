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

## Detailed Measurements

Case IDs use
`w<workers>_c<concurrent-games>_s<simulations>_b<batch-limit>_g<total-games>[_suffix]`.

| Case | Date | Players | Model | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Total games | Concurrent games per process | Generation worker processes | MCTS simulations per move | NN leaf eval batch limit | Actual NN leaf eval batch avg | Actual NN leaf eval batch max | Actual NN leaf eval batch fill | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | Generator CPU avg | Generator CPU max | Generator RSS | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `w6_c8_s16_b32_g48_leaffill` | pending | checkpoint vs checkpoint | d256-h1024-heads8-l6 | pending | pending | pending | pending | pending | 48 | 8 | 6 | 16 | 32 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | Planned NN leaf eval batch fill comparison. |
| `w6_c8_s16_b64_g48_leaffill` | pending | checkpoint vs checkpoint | d256-h1024-heads8-l6 | pending | pending | pending | pending | pending | 48 | 8 | 6 | 16 | 64 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | Planned NN leaf eval batch fill comparison. |
| `w6_c16_s16_b32_g96_leaffill` | pending | checkpoint vs checkpoint | d256-h1024-heads8-l6 | pending | pending | pending | pending | pending | 96 | 16 | 6 | 16 | 32 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | Planned NN leaf eval batch fill comparison. |
| `w6_c16_s16_b64_g96_leaffill` | pending | checkpoint vs checkpoint | d256-h1024-heads8-l6 | pending | pending | pending | pending | pending | 96 | 16 | 6 | 16 | 64 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | Planned NN leaf eval batch fill comparison. |
| `w1_c8_s128_b64_g8` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX A5000 | 9 vCPU, 50 GiB | secure | EU-SE-1 | $0.27/hr | 8 | 8 | 1 | 128 | 64 | 5.09 | 8 | 7.95% | 151.9 | 378.39 | 3.21 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | Completed. End reasons: 6 game_over, 2 max_plies. Result: black 5, white 1, draws 2. |
| `w8_c8_s128_b64_g1024` | 2026-05-18 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4090 | 8 vCPU, 46 GiB | secure | EU-RO-1 | $0.69/hr | 1024 | 8 | 8 | 128 | 64 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | Pending measurement. |
| `w6_c8_s16_b32_g48_l4` | 2026-05-14 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | L4 | 16 vCPU, 94 GiB | secure | US-MO-2 | $0.39/hr | 48 | 8 | 6 | 16 | 32 | 6.14 | 8 | 19.18% | 221.0 | 138.51 | 76.60 | 45.73% | 79.00% | 2266 MiB / 23034 MiB | 657.65% | 1201.90% | 5813 MiB |  |
| `w1_c16_s16_b32_g16` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | 16 | 16 | 1 | 16 | 32 | not recorded | not recorded | not recorded | 212.1 | 266.82 | 12.72 | 4.37% | 11.00% | 640 MiB / 20475 MiB | 102.42% | 142.00% | 995 MiB | 2026-05-13 current-code profile. Measured phase share: expand 63.81%, selection 23.38%. |
| `w2_c8_s16_b32_g16` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | 16 | 8 | 2 | 16 | 32 | not recorded | not recorded | not recorded | 212.0 | 178.58 | 18.99 | 9.95% | 29.00% | 703 MiB / 20475 MiB | 172.32% | 308.30% | 1983 MiB | 2026-05-13 current-code profile. Measured phase share: expand 64.08%, selection 22.75%. |
| `w4_c8_s16_b32_g32` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | 32 | 8 | 4 | 16 | 32 | not recorded | not recorded | not recorded | 247.0 | 193.87 | 40.77 | 24.60% | 51.00% | 1415 MiB / 20475 MiB | 356.25% | 498.10% | 3917 MiB | 2026-05-13 current-code profile. Measured phase share: expand 62.76%, selection 23.94%. |
| `w6_c8_s16_b32_g48` | 2026-05-13 | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | US | $0.20/hr | 48 | 8 | 6 | 16 | 32 | not recorded | not recorded | not recorded | 241.7 | 230.17 | 50.40 | 33.64% | 59.00% | 2089 MiB / 20475 MiB | 485.50% | 567.50% | 5825 MiB | 2026-05-13 current-code profile. Measured phase share: expand 65.81%, selection 21.63%. |
| `w6_c8_s16_b64_g48` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | not pinned; assigned US | $0.20/hr | 48 | 8 | 6 | 16 | 64 | not recorded | not recorded | not recorded | 201.1 | 194.30 | 49.68 | 34.17% | 58.00% | 2063 MiB / 20475 MiB | 447.00% | 570.70% | 5678 MiB | 18 of 48 games reached max plies. |
| `w8_c8_s16_b32_g64_6vcpu` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 6 vCPU, 31 GiB | community | `EU-RO-1` requested; assigned US | $0.20/hr | 64 | 8 | 8 | 16 | 32 | not recorded | not recorded | not recorded | 227.4 | 279.35 | 52.10 | 35.14% | 53.00% | 2816 MiB / 20475 MiB | 463.94% | 586.80% | 7555 MiB | 29 of 64 games reached max plies. |
| `w8_c8_s16_b32_g64_9vcpu` | not recorded | checkpoint vs checkpoint | d256-h1024-heads8-l6 | RTX 4000 Ada | 9 vCPU, 50 GiB | secure | `EU-RO-1` | $0.26/hr | 64 | 8 | 8 | 16 | 32 | not recorded | not recorded | not recorded | 225.9 | 147.35 | 98.10 | 54.48% | 84.00% | 2777 MiB / 20475 MiB | 643.07% | 830.40% | 7825 MiB | 28 of 64 games reached max plies. |
