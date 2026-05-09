# Shogi Decision Performance Schema

Status: open. Priority: low.

## Issue

Current shogi MCTS inference timing is passed through
`decision_usi_info_lines` as an `info string intrep_performance {...}` JSON
payload. This is useful for quick measurement, but it mixes internal performance
telemetry with USI engine info lines and relies on string parsing.

## Why It Matters

As performance measurement expands beyond the first MCTS profiling path, this
can become hard to manage:

- `InProcessShogiPlayer` reads `policy.last_performance` implicitly.
- performance data is not a structured field on `ShogiTransitionRecord`.
- parser logic depends on a string prefix and JSON payload.
- direct policy, MCTS, external engine timing, and future runtime paths may need
  different but related measurements.

## Desired Direction

Introduce a small structured field for decision-time performance, for example:

`ShogiTransitionRecord.decision_performance: ShogiDecisionPerformance | None`

Keep USI `info` lines for actual engine info. Use structured performance data
for internal timing.

## Acceptance Criteria

- MCTS timing no longer needs to be encoded as a USI info string.
- evaluation summaries read structured performance data.
- existing USI info lines remain available separately.

