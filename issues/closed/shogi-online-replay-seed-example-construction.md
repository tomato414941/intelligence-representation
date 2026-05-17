# Shogi Online Replay Seed Example Construction

Status: closed. Priority: high.

## Issue

Shogi Online Replay currently builds replay seed examples by loading the full
selected training split into `ShogiPolicyValueExample` objects before sampling.

With `qhapaq-full`, this means the run starts by turning about 4.95 million
positions into Python objects on one CPU core. The 2026-05-17 RunPod attempt
showed this clearly: the process spent more than 19 minutes in initial
CPU-only work before self-play generation or GPU training started.

This is different from replay-buffer persistence and from keeping a large
replay population in memory. The immediate problem is the construction order:

```text
full game records
-> full Python examples
-> sample
```

The desired order is:

```text
source records or indexed cache
-> sample source positions or games
-> build only the selected examples
```

## Scope

This issue is about avoiding full seed-example construction for Online Replay.

The fix should keep Qhapaq full usable as a seed source without requiring every
seed position to be materialized as a Python object at startup.

## Non-Goals

- Do not solve resume/persistence here.
- Do not introduce prioritized replay.
- Do not require the full tensor cache to be synced to every RunPod job.
- Do not change fixed offline full-training behavior.

## Acceptance Criteria

- Online Replay can start from a large game-record seed selection without
  constructing every seed example before sampling.
- Seed sampling is reproducible from the run seed.
- Metrics distinguish eligible seed examples from sampled seed examples.
- Generated experience still enters the learner data for the iteration.

## Resolution

Online Replay now treats fixed seed data as a sampling source, not as replay
state.

When the Data Selection has a tensor cache, seed sampling reads selected cache
shards directly. When it does not, seed sampling chooses source positions first
and constructs only the selected plies from game records.

Generated experience remains in the dynamic replay buffer and is mixed with the
iteration seed sample for training.

## Related

- [`shogi-online-replay-disk-backed-sampling.md`](shogi-online-replay-disk-backed-sampling.md)
- [`online-replay-buffer-persistence.md`](online-replay-buffer-persistence.md)
