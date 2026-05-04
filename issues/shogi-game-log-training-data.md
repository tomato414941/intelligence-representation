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
- record moves, players, engine settings, winner, end reason, and ply count
- write raw game log JSONL

This repository owns learning data conversion:

- read raw game log JSONL
- select policy targets by source priority
- preserve or aggregate value targets intentionally
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

## Candidate Raw Game Fields

- black player type and settings
- white player type and settings
- moves
- winner
- end reason
- ply count
- per-move actor/source
- optional per-move SFEN, if needed for conversion speed or reproducibility

## Acceptance Criteria

This issue can close when:

- `shogi-arena-agent` produces a raw game log JSONL format sufficient for
  multiple engine/model sources, and
- this repository can convert that raw log into training-ready shogi move-choice
  examples without losing source-priority and value-target decisions before the
  conversion step.
