# Shogi Value Teacher Design

Status: closed.

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

## Current Evidence

`inspect_shogi_usi_info` now reports best-line score coverage and score ranges
from stored `usi_info_lines`.

On the current shogi Experience Store, best-line scores exist for most plies:

- 482 games / 28,841 plies
- best score coverage: 26,861 plies, about 93.1%
- best cp scores: 26,224 lines, range -15347 to 19410, mean about 318 cp
- best mate scores: 637 lines, range -2 to 15

The recent `model-reached-g200-n1-mcts8-mpv3-view` Training View has similar
coverage:

- train: 20,106 / 21,487 plies, about 93.6%
- eval: 6,755 / 7,354 plies, about 91.9%

This suggests score-derived value targets are practical from the current raw
records. The remaining design work is not data availability, but sign semantics,
mate-score scaling, and how the Training View records which value teacher was
used.

## Current Route

Training now chooses value target source through Dataset Definition:

- `policy_target_source: "chosen_move"` uses the played move as a one-hot
  policy target.
- `policy_target_source: "usi_multipv"` derives policy targets from stored
  raw USI MultiPV lines.
- `value_target_source: "winner"` uses final winner-derived return targets.
- `value_target_source: "yaneuraou_best_score"` uses best-line USI score from
  stored `usi_info_lines`.
- `policy_temperature_cp` and `policy_mate_cp` record how MultiPV policy
  targets are derived.
- `score_cp_scale` records how centipawn scores are mapped with
  `tanh(score_cp / score_cp_scale)`.
- mate scores map directly to `+1.0` or `-1.0`.

This keeps the YaneuraOu value-teacher route available without making it the
only route.

`ShogiGameRecord` no longer stores derived `policy_targets`. It keeps raw
experience and raw `usi_info_lines`; `ShogiMoveChoiceExample` is where the
selected policy/value targets appear. Training metrics include the Dataset
Definition, so runs record which policy and value sources were used.

## Acceptance Criteria

This issue is closed because shogi training now has explicit value-teacher
selection through Dataset Definition:

- `value_target_source: "winner"` uses final winner-derived return targets.
- `value_target_source: "yaneuraou_best_score"` uses best-line USI score from
  stored `usi_info_lines`.
- `score_cp_scale` records how centipawn scores are mapped into `[-1.0, 1.0]`.
- mate scores map to `+1.0` or `-1.0`.
- metrics record the Dataset Definition used by the run.

Remaining questions are tracked in narrower issues:

- source-level policy/value target selection:
  [`shogi-source-target-policy-mix.md`](shogi-source-target-policy-mix.md)
- multiple evidence records for the same position:
  [`shogi-position-evidence-merge.md`](shogi-position-evidence-merge.md)
