# Shogi Generated Record Generation Latency Repro

Status: open
Priority: high

## Problem

Full generated-record creation for the next shogi training dataset can enter a
long checkpoint-vs-checkpoint self-play phase. The original concern was that no
durable output appeared until a full shard completed. That is no longer the
precise current failure mode.

Current `shogi-arena-agent` generation writes per-worker durable artifacts while
the worker is still running:

- `games.shard-NNN.jsonl`
- `games.shard-NNN.events.jsonl`
- `games.shard-NNN.progress.json`

For a single worker it writes the non-sharded equivalents:

- `games.jsonl`
- `games.events.jsonl`
- `games.progress.json`

The remaining problem is orchestration visibility. The intrep/RunPod generation
path does not make those durable progress artifacts first-class. A long source
generation can still look like a single opaque phase unless the operator knows
which shard files to inspect, and intrep's Python wrapper still captures the
arena subprocess stdout until completion.

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
sharding, output-flush, or move-generation bug. The current durable artifacts
are sufficient to diagnose this, but the orchestration layer does not surface
their paths or stream the wrapper summary/progress clearly enough.

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

Original interpretation:

- The 8-game path completed, so this does not look like a basic game-completion
  deadlock.
- Two games reached `max_plies`, so long games can delay a shard substantially.
- With 8 games taking about 6.3 minutes, a 128-game worker allocation can
  plausibly take about 100 minutes before every worker is complete and the final
  merged summary is written.
- The full 1024-game run used 8 shards of 128 games. Seeing no shard file after
  40-50 minutes would now be suspicious; current arena generation should create
  shard progress/event files early.
- The main design problem is now orchestration observability: source-level logs
  should say where durable shard progress lives, and intrep wrapper paths should
  not hide useful generation output until the subprocess exits.

## Current Investigation

Local smoke checks against current `shogi-arena-agent` confirmed:

- single-worker generation creates `games.jsonl`, `games.events.jsonl`, and
  `games.progress.json`
- multi-worker generation creates `games.shard-000.*`,
  `games.shard-001.*`, and finally merged `games.jsonl`
- shard records are appended per completed game
- shard progress/events are durable before the parent merged summary completes

Relevant current code:

- `shogi_arena_agent.generated_game_artifacts.GeneratedGameArtifacts`
  appends records and writes events/progress files
- `shogi-arena-agent/scripts/generate_shogi_games.py` still waits for worker
  subprocesses via `communicate()` before printing the aggregate summary
- `intrep.problems.shogi_policy_value.generated_game_production.run_shogi_generated_games`
  uses `subprocess.run(stdout=PIPE)`, so Python orchestration callers receive
  the arena stdout only after generation completes
- `scripts/runpod_generate_shogi_mixed_records.sh` directly invokes the arena
  script through `tee`, but it does not announce the expected progress/event
  artifact paths for each source

## Desired Shape

Generation orchestration should make durable progress discoverable while a
source is still running.

At minimum:

- each source generation start log should include the durable artifact paths
  that an operator can inspect during the run
- source logs should distinguish final merged output from per-worker shard
  output
- Python orchestration should not unnecessarily buffer useful arena stdout until
  the subprocess completes

The goal is not to invent a second record format. The arena-owned
`games*.jsonl`, `*.events.jsonl`, and `*.progress.json` files are the durable
progress surface.

## Close Condition

- RunPod mixed-record source startup logs identify the expected durable progress
  and event files for that source.
- Intrep Python generation orchestration streams or forwards arena output
  without waiting for the subprocess to complete.
- Interrupted generated-record runs leave enough discoverable durable state to
  distinguish normal long games from a broken worker.
