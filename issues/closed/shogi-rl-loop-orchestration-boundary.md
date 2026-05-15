# Shogi RL Loop Orchestration Boundary

Status: closed. Priority: medium.

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
- the generated-data training cycle script becomes a repeated production workflow
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

## Resolution

The current boundary is explicit enough to close this issue.

`intelligence-representation` owns the generated-data and Online Replay loops
in `src/intrep/problems/shogi_policy_value/generated_data_cycle.py`. Those loops
invoke `shogi-arena-agent` through `scripts/generate_shogi_games.py`, then read
the produced game-record JSONL, split or append the records, update the model,
write metrics, and promote the next checkpoint.

`shogi-arena-agent` owns runtime game generation. It constructs checkpoint,
YaneuraOu, or deterministic players; applies direct or MCTS move selection;
runs games; and writes raw game-record JSONL. It may load
`intelligence-representation` checkpoints for inference, but the RL learning
loop does not import arena internals.

The artifact boundary is now documented in `docs/learning-boundaries.md`.
Checkpoint actor provenance was addressed separately in
`closed/shogi-checkpoint-actor-provenance.md`: generated records can carry
checkpoint identity, move selector, and MCTS search settings, and Experience
Store / Training Data Bundle manifests summarize those actors.

Reopen or create a new issue if CLI/subprocess overhead becomes measured as a
material blocker, if self-play needs distributed orchestration, or if a shared
game-generation schema library becomes smaller than the current artifact
boundary.
