# Shogi Pin And Threat Input Features

Status: closed
Priority: medium

## Problem

The current shogi input representation includes cheap tactical approximations:
counterfactual removal, coarse slider blockers, drop-shadow, drop-potential, line
tokens, and pair relation edges.

It intentionally does not include deeper tactical concepts such as exact pins,
discovered attacks, threat/response tokens, mate threats, or king-safety
aggregate features.

These features may help policy/value quality, but they are closer to tactical
search hints than basic position description. Adding them too casually can make
feature generation expensive or blur the line between input representation and a
small handcrafted evaluator.

## Desired Shape

After the current representation has been exercised in training and evaluation,
decide whether to add any deeper tactical features.

Candidate feature families:

- exact pin and discovered-attack context
- threat/response tokens for forcing moves
- king-safety aggregate features
- more precise capture-to-hand danger beyond near-king pseudo-drop potential

Each candidate should have a clear cost, responsibility, and reason it is hard
for the model to infer from existing square, piece, line, and pair features.

## Non-Goals

- Do not add a tactical solver as input feature generation.
- Do not add broad handcrafted evaluation features without a concrete training
  or evaluation reason.
- Do not reopen the basic shogi position input schema issue for every tactical
  feature idea.

## Close Condition

- Decide which deeper tactical features, if any, should be added next.
- If added, define their schema and feature-generation cost.
- If deferred, record the evidence that the current representation should be
  exercised further before adding more tactical hints.

## Resolution

Do not add deeper pin/threat input features before the next shogi training run.

The current representation already includes tactical approximations such as
drop-shadow, counterfactual removal, coarse slider blockers, drop potential,
line tokens, and pair relation edges. Exact pins, discovered attacks,
threat/response tokens, mate threats, and king-safety aggregates would move the
feature generator closer to a handcrafted tactical evaluator.

Exercise the current representation in training first. Revisit deeper tactical
features only after measured failures show that the model needs a specific
concept that is too expensive or unreliable to infer from the existing square,
piece, line, and pair features.
