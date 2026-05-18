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

## Reproduction

On 2026-05-18, an 8-game checkpoint self-play repro completed on RunPod secure
RTX A5000:

- `games=8`
- `generation-worker-processes=1`
- `concurrent-games-per-process=8`
- `mcts-simulations-per-move=128`
- `nn-leaf-eval-batch-limit=64`
- `max-plies=320`
- output: `data/shogi/records/checkpoint-self-repro-20260518-g8`

Result:

- game count: 8
- end reasons: 6 `game_over`, 2 `max_plies`
- result distribution: black 5, white 1, draws 2
- average plies: 151.875
- generation wall time: 378.389 sec
- plies/sec: 3.211
- actual NN leaf eval batch size avg: 5.087
- actual NN leaf eval batch size max: 8
- actual NN leaf eval batch fill ratio avg: 0.079

Interpretation:

- The 8-game path completed, so this does not look like a basic game-completion
  deadlock.
- Two games reached `max_plies`, so long games can delay a shard substantially.
- With 8 games taking about 6.3 minutes, a 128-game shard can plausibly take
  about 100 minutes before producing a shard JSONL file.
- The full 1024-game run used 8 shards of 128 games. Seeing no shard file after
  40-50 minutes is therefore plausible without implying a hang.
- The main design problem is observability and durability granularity: progress
  is visible on stdout, but completed records are not durable until the shard
  finishes.

## Desired Shape

Generation should also expose durable progress before a full shard completes,
or use smaller shard units for long self-play runs, so interrupted runs are
diagnosable and partially useful.

## Close Condition

- Long generated-record runs no longer require waiting for a full 128-game shard
  before seeing durable progress.
- Interrupted generated-record runs leave enough durable state to distinguish
  normal long games from a broken worker.
