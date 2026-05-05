# Shogi Game Log Training Data Pipeline

Status: open.

## Issue

Self-play and engine-generated shogi data should be stored first as raw game
logs, not directly as `ShogiMoveChoiceExample` records.

`ShogiMoveChoiceExample` is a training-ready example format. Converting raw
games into it too early drops information needed to choose policy targets,
compare teacher sources, deduplicate positions, and interpret value targets.

## Responsibility Boundary

`../shogi-arena-agent` owns runtime game generation:

- run checkpoint models, USI engines, and YaneuraOu
- play engine-vs-engine games
- record plies, players, engine settings, winner, end reason, and ply count
- preserve raw per-ply USI `info ...` lines when an engine emits them
- write raw game log JSONL

This repository owns learning data conversion:

- read raw game log JSONL
- select policy targets by source priority
- preserve or aggregate value targets intentionally
- inspect raw USI info coverage before deriving training signals from it
- convert selected records into `ShogiMoveChoiceExample`
- train and evaluate checkpoints

## Why Raw Logs Matter

The same position and move can appear from multiple sources. The conversion
step needs to know which source should be preferred, for example:

```text
YaneuraOu higher nodes
> YaneuraOu lower nodes
> newer checkpoint
> older checkpoint
> random/legal baseline
```

Policy and value also need different handling. Policy can choose a single
teacher move for a position. Value should not be aggressively collapsed at raw
log time because the same position can appear in games with different later
outcomes.

USI `info ...` lines are currently source metadata, not training targets. They
should remain in raw `ShogiGameRecord` plies and be inspectable through stats.
Do not derive value targets, sample weights, or source priority from `score cp`,
`depth`, `nodes`, `pv`, or `multipv` until enough raw-log evidence exists to
justify that conversion rule.

## Candidate Raw Game Fields

- black player type and settings
- white player type and settings
- plies with side, position command, bestmove, ponder, and raw USI info lines
- winner
- end reason
- ply count
- optional per-move SFEN, if needed for conversion speed or reproducibility

## Acceptance Criteria

This issue can close when:

- `shogi-arena-agent` produces a raw game log JSONL format sufficient for
  multiple engine/model sources, and
- this repository can convert that raw log into training-ready shogi move-choice
  examples without losing source-priority and value-target decisions before the
  conversion step.
