# Shogi Generated Record Generation Latency Repro

Status: open
Priority: high

## Problem

Full generated-record creation for the next shogi training dataset can enter a
long checkpoint-vs-checkpoint self-play phase before any shard output is
durably written.

On 2026-05-18, a RunPod RTX 4090 secure run with:

- `GAMES_PER_SOURCE=1024`
- checkpoint self-play first
- `generation-worker-processes=8`
- `concurrent-games-per-process=8`
- `mcts-simulations-per-move=128`
- `nn-leaf-eval-batch-limit=64`
- `max-plies=320`

ran checkpoint self-play for more than 40 minutes without completing a 128-game
shard. The process was not obviously hung: 8 Python workers were active, each
using most of one CPU core, and GPU utilization was often around 50-70%.

This may be expected MCTS cost, but it may also hide a game-completion,
sharding, output-flush, or move-generation bug. The current output shape makes
that hard to distinguish because shard JSONL files appear only after a shard is
complete.

## Desired Shape

Before running another full 1024-games-per-source generation, establish a small
reproducible checkpoint self-play check:

- run 8 checkpoint-vs-checkpoint games with the same player/search settings
- confirm games complete
- record wall time, average plies, result distribution, and illegal-move count
- inspect whether any games are stuck near `max-plies`
- compare the small-run rate against the failed full-run observation

Generation should also expose durable progress before a full shard completes,
or use smaller shard units for long self-play runs, so interrupted runs are
diagnosable and partially useful.

## Close Condition

- An 8-game checkpoint self-play reproduction is run and interpreted.
- If the result is normal, the expected runtime for 1024 self-play games is
  recorded before retrying the full mixed generation.
- If the result is abnormal, the underlying game-generation bug is isolated.
- Long generated-record runs no longer require waiting for a full 128-game shard
  before seeing durable progress.
