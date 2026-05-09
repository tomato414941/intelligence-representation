# Shogi Position Annotation Store

Status: open.

## Problem
`ShogiGameRecord` should stay a source-side experience record. It should not
become a container for one problem's policy/value training targets.

`ShogiGameRecord.transition.policy_targets` is enough while there is one active
teacher annotation, but it will become unclear once the same position needs
multiple annotations from different teachers or search settings.

Examples:
- YaneuraOu MultiPV with different node budgets.
- A checkpoint policy with MCTS at different simulation counts.
- Future value or score annotations separate from the game outcome.

Embedding all of those directly into the transition risks mixing source
experience with derived teacher evidence.

Position annotations should be stored as evidence about a position, not as
problem-specific training targets. A problem may later derive policy/value,
transition, retrieval, or other targets from that evidence.

## Scope
- Define when shogi position annotations should be stored separately from `ShogiGameRecord`.
- Decide the minimal key for matching annotations to positions.
- Decide how a Training Data Bundle or problem should select one annotation source.
- Keep PyTorch dataset loading simple.

## Non-Goals
- Do not introduce a generic annotation store before shogi needs it.
- Do not support arbitrary teachers until at least two concrete teacher sources are in active use.
- Do not keep duplicate canonical data paths.

## Trigger
Revisit this when we need to keep more than one teacher annotation for the same
shogi position, or when storing annotation-derived targets on the source record
would hide important provenance.
