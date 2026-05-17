# Shogi Runtime Artifact Boundary

This document records the artifact boundary between `intelligence-representation`
and `shogi-arena-agent` for shogi runtime execution. It is separate from
`learning-data-boundaries.md` so the project-level learning data document can stay
domain-neutral.

## Ownership

`intelligence-representation` owns the learning loop:

- checkpoint selection
- raw game-record ingestion
- replay or fixed training-data construction
- model updates
- metrics
- checkpoint promotion

`shogi-arena-agent` owns shogi runtime execution:

- player construction
- USI engine processes
- runtime move selection
- search settings
- game execution
- raw game-record JSONL output
- player-vs-player match entrypoints
- RunPod wrappers for player-vs-player matches

## Artifact Contract

- checkpoint files flow from `intelligence-representation` to
  `shogi-arena-agent`
- shogi game-record JSONL flows from `shogi-arena-agent` back to
  `intelligence-representation`
- generated records carry actor metadata such as checkpoint identity, move
  selector, and search settings for later explanation and selection
- evaluation metrics and game records belong to the side that runs the
  evaluation

## Rule

`intelligence-representation` should pass checkpoints and read game-record
artifacts. It should not mirror shogi player-vs-player match runner CLIs.

Do not make `intelligence-representation` import `shogi-arena-agent` internals
only to run self-play. Keep the CLI/subprocess and artifact boundary until
measured overhead or schema coordination makes a smaller shared library boundary
clearly better.
