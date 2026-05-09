# Shogi USI Ponder Support

Status: open. Priority: medium.

## Issue

Floodgate-like play can benefit from thinking during the opponent's turn. In USI
terms this is ponder support: start searching a predicted opponent continuation,
reuse the result if the opponent plays the predicted move, and stop or discard it
otherwise.

This can improve GPU utilization because the model can run while waiting for the
opponent. It is more complex than root reuse because it requires managing
background search and the USI `go ponder` / `ponderhit` flow.

## Desired Direction

Do not implement this before root reuse is reliable. A small first version
should:

- support one predicted ponder move
- stop background search cleanly when the prediction misses
- reuse the search tree when `ponderhit` applies
- keep time-control handling explicit and conservative

## Risks

- Incorrect stop handling can lose on time or return a move from the wrong
  position.
- Weak policy can make predicted opponent moves miss often, wasting GPU time.
- Background GPU work can interfere with other concurrent evaluations.

## Acceptance Criteria

- USI ponder commands are handled without corrupting the current position
- predicted-hit and predicted-miss paths are both tested or smoke-checked
- wall-clock behavior is measured before enabling it by default
