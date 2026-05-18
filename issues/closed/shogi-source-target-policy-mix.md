# Shogi Source Target Policy Mix

Status: closed.

## Issue

The shogi Data Selection implementation used to choose one policy target source
and one value target source for the whole Training Data Bundle.

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

`shogi-training-data-bundle-source-mix.md` is about which records are included in a
Training Data Bundle. This issue is about how included records are converted into
policy/value training targets.

`closed/shogi-move-choice-problem-scope.md` records the rename from the old `ShogiMoveChoice` package boundary to `ShogiPolicyValue`.

If target source remains global, mixed Training Data Bundles either lose useful teacher
signals or force weak records into a target policy they cannot support.

## Resolution

Replaced the need for source-level target policy in the normal Training Data
Bundle path.

Training Data Bundle creation now applies target construction before writing
the bundle's train/eval data. The resulting `data-selection.json` points at
durable `shogi_policy_value_examples_jsonl` files:

```json
{
  "train_sources": [
    {
      "kind": "shogi_policy_value_examples_jsonl",
      "path": "train-examples.jsonl"
    }
  ]
}
```

This keeps mixed source interpretation out of training and tensor-cache
building. Source game records can still be adapted by the loader for existing
artifacts, but the preferred durable bundle boundary is constructed training
examples.

The implementation does not store derived targets in `ShogiGameRecord`.

## Non-Goals

- Do not add weighted sampling or source caps here.
- Do not reintroduce derived `policy_targets` into `ShogiGameRecord`.
- Do not create a generic annotation framework before shogi has a concrete need.

## Acceptance Criteria

- [x] make the normal Training Data Bundle path store durable policy/value
  training examples without storing derived targets in `ShogiGameRecord`.
