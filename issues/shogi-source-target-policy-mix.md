# Shogi Source Target Policy Mix

Status: open.

## Issue

The current shogi `DatasetDefinition` implementation chooses one policy target
source and one value target source for the whole Training View.

That is simple, but it cannot express cases where different record sources need
different target derivation rules.

Examples:

- YaneuraOu-annotated records may want `policy_target_source: "usi_multipv"`
  and `value_target_source: "yaneuraou_best_score"`.
- Plain self-play records may want `policy_target_source: "chosen_move"` and
  `value_target_source: "winner"`.
- Future MCTS self-play may want visit-count policy targets, if those are
  stored as raw search evidence rather than derived record fields.

## Why It Matters

Target source is different from source mix.

`shogi-training-view-source-mix.md` is about which records are included in a
Training View. This issue is about how included records are converted into
policy/value training targets.

`shogi-move-choice-problem-scope.md` tracks the broader question of whether
policy and value should remain under the `ShogiMoveChoice` problem boundary.

If target source remains global, mixed Training Views either lose useful teacher
signals or force weak records into a target policy they cannot support.

## Initial Direction

Do not implement this until a concrete mixed-source run needs it.

The likely shape is source-level target policy in the current shogi
`DatasetDefinition` implementation, or its eventual replacement:

```json
{
  "train_sources": [
    {
      "kind": "game_records_jsonl",
      "path": "teacher-games.jsonl",
      "policy_target_source": "usi_multipv",
      "value_target_source": "yaneuraou_best_score"
    },
    {
      "kind": "game_records_jsonl",
      "path": "self-play-games.jsonl",
      "policy_target_source": "chosen_move",
      "value_target_source": "winner"
    }
  ]
}
```

Before implementing, decide whether target policy belongs only at source level
or whether a Training View should also allow a default inherited by each source.

## Non-Goals

- Do not add weighted sampling or source caps here.
- Do not reintroduce derived `policy_targets` into `ShogiGameRecord`.
- Do not create a generic annotation framework before shogi has a concrete need.

## Acceptance Criteria

This issue can close when we either:

- decide the global Training View target policy is enough, or
- implement source-level target policy with tests showing different sources can
  derive different policy/value targets without storing derived targets in
  `ShogiGameRecord`.
