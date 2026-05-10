# Shogi RL Loop Orchestration Boundary

Status: open. Priority: medium.

## Issue

Future shogi RL work needs a clear boundary between the learning loop and the
arena runtime.

`intelligence-representation` should own the RL learning loop: checkpoint
selection, self-play requests, raw game record ingestion, replay or training
data construction, model updates, metrics, and checkpoint promotion.

`shogi-arena-agent` should own game generation runtime: constructing shogi
actors from checkpoints or USI engines, applying runtime search strategies such
as direct policy or MCTS, running games, and writing raw game record JSONL.

The repos may depend on each other operationally during an RL cycle, but they
should not grow a Python import cycle or duplicate each other's responsibilities.

## Current Position

Use a CLI/subprocess and artifact boundary first:

```text
intelligence-representation RL loop
  -> invoke shogi-arena-agent game generation
  -> read game record JSONL
  -> update model
  -> write next checkpoint
```

The current manual cycle script already follows this direction by invoking
`../shogi-arena-agent/scripts/generate_shogi_games.py` and then training from
the generated JSONL.

## Artifact Boundary

The first stable boundary should be:

- checkpoint files produced by `intelligence-representation`
- game record JSONL produced by `shogi-arena-agent`
- evaluation records or metrics produced by the side that runs the evaluation

Do not make `intelligence-representation` import arena internals unless the CLI
boundary becomes a measured blocker. Do not move model training, RunPod training
operations, dataset/cache construction, or checkpoint promotion into
`shogi-arena-agent`.

## Revisit Triggers

Revisit this boundary when one of these becomes true:

- self-play generation needs to run distributed or remotely
- CLI/subprocess overhead becomes material compared with game generation
- game record schema changes require coordinated versioning
- the manual RL cycle script becomes a repeated production workflow
- a shared library boundary is clearly smaller than the CLI/artifact boundary

## Acceptance Criteria

This issue can close when the shogi RL loop has an explicit orchestration path
that:

- keeps the learning loop in `intelligence-representation`
- keeps game generation runtime in `shogi-arena-agent`
- records checkpoint identity, actor settings, and search settings needed for
  replay/training decisions
- avoids Python import cycles between the two repositories
- uses a documented artifact contract for generated games and evaluation output
