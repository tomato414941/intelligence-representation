# Shogi Policy Value Data Selection Boundary

Status: closed.

## Issue

The old `ShogiPolicyValueDatasetDefinition` implementation combined multiple
responsibilities that are now separate project concepts.

It currently contains:

- Data Selection: `train_sources`, `eval_sources`, source `kind`, `path`, and
  `max_games`
- Training Example meaning: `objective`
- target policy and shaping defaults: `policy_target_source`,
  `value_target_source`, `policy_temperature_cp`, `policy_mate_cp`, and
  `score_cp_scale`
- source-level target policy overrides on individual sources

`load_shogi_policy_value_data_selection_examples()` also uses that combined structure to
select records and construct `ShogiPolicyValueExample` values in one step.

## Why It Matters

Source-level target policy now works, so the remaining concern is narrower:
record selection and `ShogiPolicyValueExample` construction still happen through
one data-selection loader.

The broader responsibility issue was closed in
`closed/training-example-responsibility-mixing.md`. This issue tracks the shogi
policy-value concrete case.

## Related Issue

`closed/shogi-source-target-policy-mix.md` resolved the target-policy mix
problem by allowing source-level target policy overrides. This issue is broader:
it asked whether the current shogi data-selection implementation should be split
or renamed now that `Data Selection` is the formal inclusion-boundary term.

`closed/shogi-move-choice-problem-scope.md` tracks that the old
`ShogiMoveChoice` package boundary has been renamed to `ShogiPolicyValue`.

## Direction

The implementation was renamed from dataset-definition language to data-selection
language:

- `dataset_definition.py` -> `data_selection.py`
- `ShogiPolicyValueDatasetDefinition` -> `ShogiPolicyValueDataSelection`
- `ShogiPolicyValueDatasetSource` -> `ShogiPolicyValueDataSelectionSource`
- CLI `--dataset-definition` -> `--data-selection`
- generated `dataset.json` -> `data-selection.json`

Target policy remains in the same data-selection file for now. That is
intentional KISS: current runs need one compact file that says what records are
included and what target policy is used when deriving policy/value examples.

No compatibility alias was kept.

## Acceptance Criteria

- [x] Decide whether the old `ShogiPolicyValueDatasetDefinition` name should
  be replaced.
- [x] Decide whether target policy belongs globally, per source, or both with a
  default/inheritance rule.
- [x] Decide whether record selection and `ShogiPolicyValueExample` construction
  should remain one loader step or become separate steps.
- [x] Update code and tests only after the boundary decision is needed by a concrete
  mixed-source run.

Resolution: keep record selection and example construction in one loader for
now. Split only when a concrete run needs independent reuse of selected records
before policy/value example construction.

## Non-Goals

- immediate code refactor
- generic data-selection framework
- backward-compatible alias layer
