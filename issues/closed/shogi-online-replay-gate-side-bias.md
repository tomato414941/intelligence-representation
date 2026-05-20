# Shogi Online Replay Gate Side Bias

Status: closed
Priority: medium

## Problem

The online replay gate alternates sides, but recent gate results still show a
large side split.

In the 2026-05-17 RunPod online replay run, the iteration 3 gate was 16-14-2
overall, but the side breakdown was:

- player A as black: 12-2
- player A as white: 4-12

This does not prove a bug by itself, but it matters because the gate is used to
decide whether the current checkpoint should become the next data generator.

## Desired Shape

Gate results should be interpreted with side breakdown visible.

If side advantage dominates the result, the gate should not be treated as a
clean strength signal without qualification.

Possible policies include:

- require both sides to be non-catastrophic
- report side-adjusted outcome separately from raw score
- increase game count or use paired openings only if needed

Do not add a complex rating system before the bias is measured across more runs.

## Close Condition

- Gate summaries keep side-specific wins/losses visible.
- Online replay continuation logic has an explicit policy for side-skewed gate
  results.
- A future experiment can say whether side bias was ignored, tolerated, or used
  in the decision.

## Resolution

2026-05-20:

- Online replay gate results now include `side_breakdown` computed from the
  durable `generator-gate-games.jsonl` game records.
- The breakdown records candidate results as black, as white, and unknown-side
  records if actor identity cannot be matched.
- Side skew is recorded for interpretation but is not currently used as a stop
  condition. The gate remains a degradation guard based on the overall
  `decision`.
- Future experiment summaries can state that side bias was recorded and
  tolerated, not ignored or used for stopping.
