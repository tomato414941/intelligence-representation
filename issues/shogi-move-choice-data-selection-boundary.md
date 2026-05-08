# Shogi Move Choice Data Selection Boundary

Status: open.

## Issue

The current `ShogiMoveChoiceDatasetDefinition` implementation combines multiple
responsibilities that are now separate project concepts.

It currently contains:

- Data Selection: `train_sources`, `eval_sources`, source `kind`, `path`, and
  `max_games`
- Training Example meaning: `objective`
- target policy and shaping defaults: `policy_target_source`,
  `value_target_source`, `policy_temperature_cp`, `policy_mate_cp`, and
  `score_cp_scale`
- source-level target policy overrides on individual sources

`load_shogi_move_choice_dataset_examples()` also uses that combined structure to
select records and construct `ShogiPolicyValueExample` values in one step.

## Why It Matters

Source-level target policy now works, so the remaining concern is narrower:
record selection and `ShogiPolicyValueExample` construction still happen through
one `DatasetDefinition` loader.

The broader responsibility issue was closed in
`closed/training-example-responsibility-mixing.md`. This issue tracks the shogi
move-choice concrete case.

## Related Issue

`closed/shogi-source-target-policy-mix.md` resolved the target-policy mix
problem by allowing source-level target policy overrides. This issue is broader:
it asks whether the current shogi `DatasetDefinition` implementation should be
split or renamed now that `Data Selection` is the formal inclusion-boundary
term.

`shogi-move-choice-problem-scope.md` tracks whether `ShogiMoveChoice` is the
right problem boundary for policy/value/data-construction responsibilities.

## Direction

Do not refactor immediately. Source-level target policy has been implemented,
and a broader problem-scope question remains open.

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
- [x] Decide whether target policy belongs globally, per source, or both with a
  default/inheritance rule.
- Decide whether record selection and `ShogiPolicyValueExample` construction
  should remain one loader step or become separate steps.
- [x] Update code and tests only after the boundary decision is needed by a concrete
  mixed-source run.

## Non-Goals

- immediate code refactor
- generic data-selection framework
- backward-compatible alias layer
