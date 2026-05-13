# Shogi MCTS Runtime Bottleneck Classification

Status: closed. Priority: high.

## Issue

Shogi MCTS with batched neural evaluation is not always dominated by GPU model
time in arena-like workloads. The next step is to classify the current runtime
bottleneck using the existing per-move performance and phase timing metrics.

This is a diagnosis issue, not an open-ended optimization issue.

The 2026-05-10 RunPod evaluation of the promoted
`d256-h1024-heads8-l6-shogi` checkpoint against YaneuraOu MaterialLv1
`go nodes 1` used:

```text
MCTS simulations: 4096
evaluation_batch_size: 64
device: cuda
GPU: RTX 4090
torch: 2.4.1+cu124
games: 4
max plies: 80
```

It produced:

```text
request_wall_time_sec_avg: 10.887
request_wall_time_sec_p95: 22.631
request_wall_time_sec_max: 33.782
model_wall_time_sec_avg: 3.100
non_model_wall_time_sec_avg: 7.788
model_call_count_avg: 115.779
```

This exceeded the 10-second move budget on average. Since model wall time was
only about 3.1s, increasing GPU batching alone was unlikely to solve that
particular measured case.

## Why It Matters

The short deterministic MCTS grids made `MCTS4096` look viable, but a more
arena-like YaneuraOu workload had much higher CPU-side overhead and tail
latency.

Candidate bottleneck classes:

- legal move generation
- board copy / push / pop behavior
- MCTS tree traversal and child selection
- Python dict/list overhead in node expansion and backup
- larger legal move sets in tactical or late-game positions
- batches not filling ideally; for the historical MCTS4096/batch64 run,
  `4096 / 64` suggests 64 model calls, but the measured average was about 116
  calls

This is a runtime/search issue, not primarily a model-training issue.

## Additional Observation: Self-Play Generation

The 2026-05-12 RunPod Online Replay confirmation attempted self-play game
generation with:

```text
GPU: RTX 5090
torch: 2.8.0+cu128
checkpoint: d256-h1024-heads8-l6-shogi
games: 16
concurrent_games_per_process: 16
MCTS simulations: 128
evaluation_batch_size: 64
max plies: 320
board backend: cshogi
```

After about 15 minutes, cycle 1 was still inside `generate_shogi_games.py`.
Spot checks showed low GPU utilization around 2-3% while the Python generation
process used roughly one CPU core. Later self-play measurements showed that
process-level generation workers improved throughput materially, so self-play
generation has its own parallelism path. This issue focuses on classifying the
one-game MCTS runtime bottleneck.

## Scope

In scope:

- inspect or rerun one-game MCTS measurements with phase timing
- classify the current bottleneck as model time, legal moves, board copy,
  selection, expansion, backup, batch fill/model calls, or unattributed time
- create narrower follow-up issues if a specific optimization is justified

Out of scope:

- optimizing MCTS internals directly
- choosing a permanent arena default
- self-play worker scaling
- training-data loading or tensor-cache work

## Current Measurement Hook

`shogi-arena-agent` already records the needed timing fields:

- `model_wall_time_sec`
- `non_model_wall_time_sec`
- `actual_nn_leaf_eval_batch_size_avg`
- `actual_nn_leaf_eval_batch_size_max`
- `actual_nn_leaf_eval_batch_count`
- `phase_wall_time_sec`

Known phase names include:

- `position_parse`
- `legal_moves`
- `board_copy`
- `selection`
- `batch_build`
- `expand`
- `backup`
- `unattributed` / `unattributed_wait`

## Acceptance Criteria

This issue can close when:

- a representative one-game MCTS measurement records model, non-model, actual
  batch, and, when available, phase timing fields
- the dominant bottleneck class is identified from those measurements
- any required optimization work is split into narrower follow-up issues
- the diagnosis is recorded in this issue's resolution or in
  `docs/inference-performance.md`

## Resolution

Closed on 2026-05-13.

A RunPod one-game check classified the current representative MCTS2048/batch64
case as model-call/model-wall dominated, not CPU-phase dominated:

```text
GPU: RTX 4000 Ada Generation
vCPU/RAM: 6 vCPU, 31 GiB
template: runpod-torch-v280
torch: 2.8.0+cu128
checkpoint: d256-h1024-heads8-l6-shogi
opponent: YaneuraOu MaterialLv1, go nodes 1
board backend: cshogi
MCTS simulations: 2048
NN leaf eval batch limit: 64
move time limit: 9.0s
games: 1
max plies: 320
result: 0-1, game_over, 50 plies
```

Measured inference summary:

```text
request_wall_time_sec_avg: 4.576
request_wall_time_sec_p95: 8.776
request_wall_time_sec_max: 9.107
model_wall_time_sec_avg: 3.840
non_model_wall_time_sec_avg: 0.737
model_call_count_avg: 51.4
actual_nn_leaf_eval_batch_size_avg: 44.34
actual_nn_leaf_eval_batch_size_max: 64
GPU util avg/max: 6.89% / 17.0%
CPU util avg/max: 20.46% / 24.69%
```

The model wall time was about 84% of average request wall time. The non-model
time was about 16%, so this representative MCTS2048 run does not justify a
narrow CPU-phase optimization issue by itself.

`phase_wall_time_sec` was emitted as an empty object on this one-game path, so
this run did not classify the non-model component into legal moves, board copy,
selection, expansion, or backup. Because the non-model component was not the
dominant cost in this representative run, no follow-up phase-instrumentation
issue is opened now.
