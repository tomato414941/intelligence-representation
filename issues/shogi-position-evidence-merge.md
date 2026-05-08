# Shogi Position Evidence Merge

Status: open. Priority: low.

## Issue

The same shogi position can appear with multiple pieces of evidence from
different sources.

Examples include:

- YaneuraOu score or MultiPV at different node counts
- model MCTS visit counts
- the move actually played by an actor
- final game outcome
- repeated appearances of the same position in different games

The raw Experience Store should not collapse these records. However, a Training
View or derived dataset may eventually need a policy for how to use multiple
evidence records attached to the same position.

## Why It Matters

This is different from source mix. Source mix decides which games or sources
enter a Training Data Bundle. Position evidence merge decides what to do when multiple
included records describe the same position.

This is also different from target-source policy. Target-source policy decides
how a record source becomes policy/value targets. Position evidence merge
decides whether multiple target/evidence candidates for the same position are
kept separately, prioritized, averaged, weighted, or deduplicated.

If this remains implicit, the project may accidentally overweight repeated
openings, mix weak and strong teacher signals without visibility, or discard
useful higher-quality annotations.

## Initial Policy

Do not merge evidence in the raw Experience Store.

Current Training Data Bundles may keep repeated positions as repeated samples. This is
acceptable while duplicate/evidence effects are visible through manifest stats
and no concrete run needs a merge policy.

Do not implement a generic annotation framework yet. Keep this issue focused on
shogi positions and only implement a merge policy when a concrete Training Data Bundle
or evaluation run needs it.

Likely choices to evaluate later:

- keep repeated positions as repeated training samples
- choose one evidence source by priority
- aggregate policy targets from multiple teachers
- aggregate value targets from multiple evaluations
- keep multiple evidence rows but expose duplicate/evidence stats

## Non-Goals

- choosing actor-pair source ratios for a Training Data Bundle
- checkpoint actor provenance
- train/eval overlap exclusion
- tensorized Training Data Bundle cache
- storing derived policy/value targets in `ShogiGameRecord`

## Acceptance Criteria

This issue can close when the project either:

- decides repeated position evidence should intentionally remain unmerged for
  current shogi training, with stats sufficient to see the effect, or
- implements a narrow position-evidence merge policy for shogi Training Data Bundles
  or derived datasets, with tests showing how conflicting policy/value evidence
  is handled.
