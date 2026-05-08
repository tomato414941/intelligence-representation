# Shogi Position Annotation Store

Status: open.

## Problem
`ShogiGameRecord.transition.policy_targets` is enough while there is one active teacher annotation, but it will become unclear once the same position needs multiple annotations from different teachers or search settings.

Examples:
- YaneuraOu MultiPV with different node budgets.
- A checkpoint policy with MCTS at different simulation counts.
- Future value or score annotations separate from the game outcome.

Embedding all of those directly into the transition risks mixing the source position record with derived teacher evaluations.

## Scope
- Define when shogi position annotations should be stored separately from `ShogiGameRecord`.
- Decide the minimal key for matching annotations to positions.
- Decide how a Training Data Bundle should select one annotation source.
- Keep PyTorch dataset loading simple.

## Non-Goals
- Do not introduce a generic annotation store before shogi needs it.
- Do not support arbitrary teachers until at least two concrete teacher sources are in active use.
- Do not keep duplicate canonical data paths.

## Trigger
Revisit this when we need to keep more than one teacher annotation for the same shogi position, or when overwriting `policy_targets` would hide important provenance.
