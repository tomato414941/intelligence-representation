# Shogi MCTS CPU Overhead

Status: open. Priority: high.

## Issue

Shogi MCTS with batched neural evaluation is no longer dominated by GPU model
time in arena-like workloads. The current bottleneck is CPU-side search
overhead.

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
only about 3.1s, increasing GPU batching alone is unlikely to solve the budget
problem.

## Why It Matters

The short deterministic MCTS grids made `MCTS4096` look viable, but a more
arena-like YaneuraOu workload had much higher CPU-side overhead and tail
latency.

Likely contributors:

- `python-shogi` legal move generation
- board copy / push / pop behavior
- MCTS tree traversal and child selection
- Python dict/list overhead in node expansion and backup
- larger legal move sets in tactical or late-game positions
- batches not filling ideally; `4096 / 64` suggests 64 model calls, but the
  measured average was about 116 calls

This is a runtime/search issue, not primarily a model-training issue.

## Current Position

Do not treat `MCTS4096, evaluation_batch_size=64` as a safe 10-second arena
setting for this checkpoint.

Use lower `simulation_count` values for the next arena-like measurements before
optimizing internals.

Candidate next measurements:

```text
MCTS2048, evaluation_batch_size=64
MCTS2048, evaluation_batch_size=128
MCTS1024, evaluation_batch_size=64
```

## Candidate Directions

Measure before refactoring. If lower simulation counts are not enough, consider
targeted runtime improvements:

- profile per-move CPU time inside MCTS selection, expansion, legal move
  generation, and backup
- reduce board copying or repeated legal move generation
- cache per-position legal moves and priors where safe
- improve leaf collection so batches fill more consistently
- move hot MCTS bookkeeping out of Python only if profiling justifies it
- add a time-budgeted MCTS mode that stops before the arena wall-clock limit

## Acceptance Criteria

This issue can close when either:

- an arena-like YaneuraOu workload has a documented MCTS setting that stays
  within the 10-second move budget with acceptable tail latency, or
- profiling identifies and fixes the CPU-side bottleneck enough for the intended
  simulation count to fit within budget.
