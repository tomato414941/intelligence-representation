# Shogi Move Choice Data Selection Boundary

Status: open.

## Issue

The current `ShogiMoveChoiceDatasetDefinition` implementation combines multiple
responsibilities that are now separate project concepts.

It currently contains:

- Data Selection: `train_sources`, `eval_sources`, source `kind`, `path`, and
  `max_games`
- Training Example meaning: `objective`
- target policy and shaping: `policy_target_source`, `value_target_source`,
  `policy_temperature_cp`, `policy_mate_cp`, and `score_cp_scale`

`load_shogi_move_choice_dataset_examples()` also uses that combined structure to
select records and construct `ShogiMoveChoiceExample` values in one step.

## Why It Matters

This is workable for simple runs, but it becomes awkward when different sources
need different target policies.

Examples:

- YaneuraOu-annotated records may want `usi_multipv` policy targets and
  `yaneuraou_best_score` value targets.
- Plain self-play records may want `chosen_move` policy targets and `winner`
  value targets.

The broader responsibility issue is tracked in
`training-example-responsibility-mixing.md`. This issue tracks the shogi move
choice concrete case.

## Related Issue

`shogi-source-target-policy-mix.md` tracks the target-policy mix problem. This
issue is broader: it asks whether the current shogi `DatasetDefinition`
implementation should be split or renamed now that `Data Selection` is the
formal inclusion-boundary term.

## Direction

Do not refactor immediately. The first concrete refactor should likely happen
when a mixed-source shogi run requires source-level target policy.

Possible future shape:

- `ShogiMoveChoiceDataSelection`: train/eval source inclusion
- `ShogiMoveChoiceSourceSelection`: source kind/path/max_games and optional
  source-local target policy
- `ShogiMoveChoiceTrainingExampleSpec`: objective and default target policy /
  shaping settings

Avoid compatibility aliases when this is eventually renamed or split.

## Acceptance Criteria

- Decide whether the current `ShogiMoveChoiceDatasetDefinition` name should be
  replaced.
- Decide whether target policy belongs globally, per source, or both with a
  default/inheritance rule.
- Decide whether record selection and `ShogiMoveChoiceExample` construction
  should remain one loader step or become separate steps.
- Update code and tests only after the boundary decision is needed by a concrete
  mixed-source run.

## Non-Goals

- immediate code refactor
- generic data-selection framework
- backward-compatible alias layer
