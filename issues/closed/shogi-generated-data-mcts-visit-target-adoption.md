# Shogi Generated Data MCTS Visit Target Adoption

Status: closed
Priority: high

## Problem

`intelligence-representation` can construct shogi policy targets from MCTS root
child visit counts, and Online Replay already uses `mcts_visit_counts` for
generated games.

The generated-data training cycle still constructs generated examples with
`policy_target_construction="chosen_move"`. That can discard most of the MCTS
search signal when generated checkpoint games are used as supervised training
data.

## Desired Shape

Generated checkpoint MCTS records should use MCTS visit-count policy targets
when the records contain the required search evidence.

The boundary should stay explicit:

- source records preserve raw MCTS search evidence
- target construction chooses `mcts_visit_counts`
- training examples receive normalized policy targets

Do not make non-MCTS sources pretend to have visit-count targets.

## Close Condition

- The generated-data training path can request `mcts_visit_counts` instead of
  `chosen_move` for checkpoint MCTS records.
- Tests cover generated-game target construction using visit counts.
- Missing visit-count evidence fails clearly or falls back only when explicitly
  configured.

## Investigation Notes

Current state:

- `shogi-arena-agent` records MCTS root child visit counts under
  `decision_telemetry.search_evidence.mcts_root_child_visit_counts`.
- `intelligence-representation` can normalize that evidence through
  `policy_target_construction="mcts_visit_counts"`.
- Online Replay generated games already use `mcts_visit_counts`.
- The fixed generated-data training cycle converted generated games into
  `shogi_policy_value_examples_jsonl` with
  `policy_target_construction="chosen_move"`. That cycle path was later
  retired.

The adoption gap was therefore narrow:

```text
src/intrep/problems/shogi_policy_value/generated_data_cycle.py
  _write_examples(...)
    load_shogi_move_policy_value_examples_from_game_records_jsonl(
      policy_target_construction="chosen_move",
      value_target_construction="winner",
    )
```

One subtle issue remains: tensorization intentionally falls back to a one-hot
chosen-move target when `example.policy_targets is None`. That is useful for
ordinary chosen-move datasets, but it can hide missing MCTS search evidence if a
generated checkpoint dataset was expected to use visit counts.

The implementation should make that boundary explicit rather than relying on an
implicit fallback.

## Close Note

Closed after target construction was made explicit on generated-data training
cycles and generated experience sources.

`ShogiMovePolicyValueExample` now records `policy_target_source` and
`value_target_source`, and tensorization rejects missing policy targets for
non-`chosen_move` policy sources. Checkpoint self-play generated data defaults
to `mcts_visit_counts`; USI mixed generated sources explicitly use
`chosen_move`.
