# Shogi Game Log Training Data Pipeline

Status: closed.

## Issue

Self-play and engine-generated shogi data should be stored first as raw game
logs, not directly as `ShogiMoveChoiceExample` records.

`ShogiMoveChoiceExample` is a training-ready example format. Converting raw
games into it too early drops information needed to choose policy targets,
compare teacher sources, deduplicate positions, and interpret value targets.

This does not mean learning must only consume `ShogiGameRecord`. This project
should allow different learning forms: image-to-text generation, classification,
move-choice prediction, value learning, and future reinforcement-learning views
may all use different training inputs. The distinction here is only between
source records that preserve what happened and derived training examples or
caches that serve a particular objective.

Use the dataset/run boundary in
[`docs/learning-boundaries.md`](../docs/learning-boundaries.md): a shogi
training dataset is not the same thing as one generated game-log or example-cache
run.

## Responsibility Boundary

`../shogi-arena-agent` owns runtime game generation:

- run checkpoint models, USI engines, and YaneuraOu
- play engine-vs-engine games
- record transitions, actors, engine settings, winner, end reason, and ply count
- preserve raw per-transition USI `info ...` lines when an engine emits them
- write raw game log JSONL

This repository owns learning data conversion:

- read raw game log JSONL
- select policy targets by source priority
- preserve or aggregate value targets intentionally
- inspect raw USI info coverage before deriving training signals from it
- convert selected records into `ShogiMoveChoiceExample`
- treat `ShogiMoveChoiceExample` JSONL as a derived training input or cache, not
  the only legitimate learning input format
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
should remain in raw `ShogiGameRecord` transitions and be inspectable through stats.
Do not derive value targets, sample weights, or source priority from `score cp`,
`depth`, `nodes`, `pv`, or `multipv` until enough raw-log evidence exists to
justify that conversion rule.

## Candidate Raw Game Fields

- black actor type and settings
- white actor type and settings
- transitions with state, legal actions, chosen action, next state, reward,
  done flag, and raw USI info lines
- winner
- end reason
- ply count
- optional per-move SFEN, if needed for conversion speed or reproducibility

## Acceptance Criteria

This issue is closed because:

- `shogi-arena-agent` produces a raw game log JSONL format sufficient for
  multiple engine/model sources,
- this repository can convert that raw log into training-ready shogi move-choice
  examples without treating those examples as source records,
- shogi Experience Stores and Training Data Bundles preserve the raw-record to
  derived-example boundary, and
- Data Selection records the selected policy and value target sources used
  during conversion.

Remaining source mix, target policy mix, repeated-position evidence, forgetting,
and checkpoint provenance questions are tracked in narrower issues:

- [`shogi-training-data-bundle-source-mix.md`](shogi-training-data-bundle-source-mix.md)
- [`shogi-source-target-policy-mix.md`](shogi-source-target-policy-mix.md)
- [`shogi-position-evidence-merge.md`](shogi-position-evidence-merge.md)
- [`shogi-training-data-bundle-forgetting-policy.md`](shogi-training-data-bundle-forgetting-policy.md)
- [`shogi-checkpoint-actor-provenance.md`](shogi-checkpoint-actor-provenance.md)
