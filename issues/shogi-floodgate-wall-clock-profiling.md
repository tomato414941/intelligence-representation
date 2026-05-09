# Shogi Floodgate Wall-Clock Profiling

Status: open. Priority: medium.

## Issue

GPU efficiency for Floodgate-like play must be judged by wall-clock behavior, not
only by training throughput or raw model FLOPs. A model can use CUDA and still be
bad for play if per-move latency is dominated by CPU search, legal move
generation, tiny batch inference, or CPU/GPU synchronization.

Without per-move profiling, it is unclear whether to optimize MCTS batching,
model shape, dtype, legal move generation, or search parameters.

## Desired Direction

Add a small measurement path for shogi checkpoint play that records:

- per-move wall time
- p95 move time
- model evaluation call count
- effective model batch size, if batching exists
- simulations per second
- basic GPU identity and utilization evidence

This should be measurement-only. It should not change playing strength or search
behavior.

## Acceptance Criteria

- a checkpoint-vs-checkpoint or checkpoint-vs-engine evaluation can emit a small
  wall-clock summary
- the summary is enough to tell whether GPU inference or CPU-side search is the
  likely bottleneck
- compute-cost notes can reference the measured wall-clock result when needed
