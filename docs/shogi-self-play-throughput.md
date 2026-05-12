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
parallel-games=4
simulations=16
evaluation-batch-size=32
```

## 2026-05-12 RTX 4000 Ada GPU Utilization Check

Context:

| Item | Value |
| --- | --- |
| GPU | RunPod RTX 4000 Ada Generation |
| RunPod rate | TBD |
| torch/CUDA | RunPod PyTorch 2.8 template, CUDA available |
| model | `d256-h1024-heads8-l6-shogi` |
| board backend | `cshogi` |
| max plies | 320 |
| player profile | checkpoint self-play MCTS on both sides |
| total job runtime | TBD |
| remote workload runtime | TBD |
| estimated total cost | TBD |

Measured results:

| Case | Total games | Concurrent games per process | MCTS simulations per move | NN leaf eval batch limit | Avg plies | Wall sec | Plies/sec | GPU util avg | GPU util max | GPU memory used | CPU observation | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `p4_s16_b32` | 4 | 4 | 16 | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `p8_s16_b32` | 8 | 8 | 16 | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `p16_s16_b32` | 16 | 16 | 16 | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Notes

- Wall time is sensitive to game length. Use `plies/sec` when comparing
  throughput across self-play settings.
- `runs/` is disposable. Measurements that should survive must be summarized in
  this document instead of relying on run-local paths.
- The measured grid used RTX 4000 Ada. Do not assume the same ranking holds on
  RTX 4090 or RTX 5090 without measuring.
- The run did not record GPU utilization over time. It confirms CUDA execution,
  but not whether the GPU was saturated.
- Increasing simulations from 16 to 32 reduced throughput in this grid.
- Process-level game parallelism is still unresolved. Current `parallel-games`
  batches multiple active games inside one Python process.
- Current measurements do not use in-tree leaf selection parallelism; batching
  comes from multiple active games, not multiple pending leaves from one MCTS
  tree.
