# Shogi Value Teacher Design

Status: open.

## Issue

Shogi value targets currently come from the final game winner only.

`ShogiMoveChoiceExample.value_target` is derived during conversion from
`ShogiGameRecord.winner`: positions where the side to move later wins get
`+1.0`, and positions where the side to move later loses get `-1.0`. Positions
from games without a winner have no value target.

This is a return target, not an explicit teacher evaluation of the position.
It is useful, but it is too coarse to be the only value teacher as the project
moves toward RL-style generated experience and search.

## Why It Matters

Policy targets can already come from richer teacher signals such as YaneuraOu
MultiPV distributions. Value targets do not yet have an equivalent path.

This makes value learning hard to interpret:

- early and middle-game positions get noisy final-outcome labels
- model-vs-YaneuraOu all-loss games can bias value toward "model-side loses"
- MCTS uses the model value, so poor or overconfident value can hurt play even
  when policy dataset metrics improve
- YaneuraOu `score cp` / `score mate` lines are preserved in raw USI info, but
  they are not yet converted into value targets

## Scope

- Decide when winner-derived return targets should be used.
- Decide whether YaneuraOu score or mate output should produce value targets.
- Define how score-derived value should be scaled into `[-1.0, 1.0]`.
- Decide whether score-derived value belongs directly in `ShogiGameRecord` or
  in a separate annotation path.
- Ensure Training View can make the chosen value-teacher source explicit.

## Non-Goals

- Do not introduce a generic annotation framework before shogi needs it.
- Do not support arbitrary teacher engines in the first pass.
- Do not silently mix winner-derived and score-derived value targets without
  recording which source was used.

## Acceptance Criteria

This issue can close when shogi training has an explicit value-teacher policy:
winner-derived return targets, score-derived teacher targets, or a documented
choice between them.

The implementation should make it clear which value target source a Training
View used, and should keep policy-target handling separate from value-target
handling.
