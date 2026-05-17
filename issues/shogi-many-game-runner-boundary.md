# Shogi Many-Game Runner Boundary

## Problem

Shogi training-data generation and player-match evaluation both run many shogi
games, but their orchestration has split into separate paths.

The shared low-level operation is `play_shogi_game(...)` in
`shogi-arena-agent`. Above that, the code diverges:

- `generate_shogi_games.py` / `shogi_generation.py`
  - fixed black/white actors
  - writes `ShogiGameRecord` JSONL
  - supports `generation-worker-processes`
  - supports `concurrent-games-per-process` for checkpoint-vs-checkpoint MCTS
  - emits generation progress for batched checkpoint MCTS
- `evaluate_shogi_players.py`
  - player A / player B with alternating sides
  - writes `ShogiGameRecord` JSONL and a match summary
  - currently runs games serially
  - owns match-specific aggregation through `match_evaluation.py`

This means the project already has useful many-game machinery, but it is only
available to generation. Longer player matches, such as the 2026-05-17
100-game YaneuraOu MaterialLv1 `go nodes 1000` check, still run as one serial
process even though each game is mostly independent.

## Current Evidence

- `scripts/evaluate_shogi_players.py` loops directly over `range(args.games)`
  and calls `play_shogi_game(...)` one game at a time.
- `scripts/generate_shogi_games.py` already shards work with subprocesses via
  `--generation-worker-processes`.
- `src/shogi_arena_agent/shogi_generation.py` has a second many-game path for
  checkpoint-vs-checkpoint batched MCTS.
- `scripts/runpod_shogi_player_matches.sh` syncs and calls
  `scripts/evaluate_shogi_players.py` directly, so player-match parallelism
  would also affect the RunPod CLI contract and synced files.
- `match_evaluation.py` only summarizes completed records and side assignments;
  it does not care whether records were produced serially or by shards.

## Desired Shape

There should be one clear boundary for running many games.

The likely shape is:

- keep `play_shogi_game(...)` as the one-game primitive
- introduce a reusable many-game runner for:
  - fixed-side generation
  - alternating-side player matches
  - optional subprocess sharding
  - progress reporting
  - deterministic side assignment and seed assignment
- keep match-specific aggregation in `match_evaluation.py`
- keep training-data-specific policy in the generation/training code

The runner should not turn generation records and match summaries into one
schema. They share game execution, not downstream meaning.

## Design Questions

- Should checkpoint-vs-checkpoint in-process batched MCTS remain a specialized
  generation path, or become an optimization under the shared runner?
- How should USI engine processes be isolated per worker so that subprocess
  sharding is safe?
- Should player-match progress and generation progress use the same progress
  payload shape?
- Should `runpod_shogi_player_matches.sh` continue to call one CLI, or should
  RunPod jobs call a narrower many-game runner entrypoint?

## Close Conditions

- Player-match evaluation has a deliberate many-game boundary instead of an
  ad hoc serial loop.
- The boundary supports player-match subprocess sharding or explicitly records
  why that is not supported.
- Existing generation behavior still has a clear owner.
- The RunPod player-match path uses the same boundary as local evaluation.
