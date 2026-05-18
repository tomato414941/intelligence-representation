# Shogi MCTS Visit Count Policy Target

Status: closed
Priority: high

## Problem

Generated shogi records can be produced by MCTS, but current policy/value
training does not use the MCTS root visit distribution as a policy target.

The current effective flow is:

```text
MCTS search
-> choose one move
-> train the chosen move as the policy label
```

This is weaker than the AlphaZero-style self-play signal:

```text
MCTS search
-> record root child visit counts
-> train the visit-count distribution as the policy target
```

`shogi-arena-agent` already computes visit-count policy targets internally, but
the generated `ShogiGameRecord` does not preserve the underlying MCTS search
evidence in a form that `intelligence-representation` can use.

## Desired Shape

Do not store derived `policy_targets` directly in `ShogiGameRecord`.

Instead:

- `shogi-arena-agent` records raw MCTS root search evidence in
  `decision_telemetry` for each MCTS-selected move.
- `intelligence-representation` adds a shogi policy target construction that
  derives policy targets from that search evidence.
- Data Selection can explicitly request MCTS visit-count policy targets for
  sources that contain the evidence.

## Non-Goals

- Do not add a generic annotation framework.
- Do not merge repeated-position evidence.
- Do not require external USI engines to provide MCTS visit counts.
- Do not make generated records automatically preferred over Qhapaq records.

## Close Condition

- Generated checkpoint MCTS games can preserve root child visit counts.
- `intelligence-representation` can derive normalized policy targets from those
  counts.
- Tests cover round-trip record JSON and target construction.

## Resolution

`shogi-arena-agent` records MCTS root child visit counts under
`decision_telemetry.search_evidence.mcts_root_child_visit_counts`.

`intelligence-representation` reads that raw search evidence and supports
`target_construction.policy = "mcts_visit_counts"`, which normalizes positive
legal-move visit counts into policy targets.
