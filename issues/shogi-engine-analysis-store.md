# Shogi Engine Analysis Store

Status: open.

## Problem
`ShogiGameRecord` should stay a source-side experience record. It should not
become a container for one problem's policy/value training targets.

For the current need, the narrow concept is shogi engine analysis: how a shogi
engine analyzes a shogi position.

`ShogiGameRecord.transition.usi_info_lines` is enough while each game record has
one active engine analysis, but it will become unclear once the same position
needs multiple analyses from different engines or search settings.

Examples:
- YaneuraOu MultiPV with different node budgets.
- Another USI engine with different search settings.
- Future value or score analyses separate from the game outcome.

Embedding all of those directly into the transition risks mixing source
experience with derived teacher evidence.

Engine analyses should be stored as analysis of a position, not as
problem-specific training targets. A problem may later derive policy/value,
transition, retrieval, or other targets from that analysis.

## Scope
- Define when shogi engine analyses should be stored separately from `ShogiGameRecord`.
- Decide the minimal key for matching engine analyses to positions.
- Decide how a Training Data Bundle or problem should select one engine analysis source.
- Keep PyTorch dataset loading simple.

## Non-Goals
- Do not introduce a generic annotation or evidence store before shogi needs it.
- Do not support arbitrary teachers, model MCTS traces, or non-engine analysis
  until a concrete need exists.
- Do not keep duplicate canonical data paths.

## Trigger
Revisit this when we need to keep more than one engine analysis for the same
shogi position, or when storing analysis-derived targets on the source record
would hide important provenance.

## Current Step

`ShogiEngineAnalysis` exists as a narrow JSONL schema for shogi-engine analysis
of shogi positions. It is intentionally not a generic annotation/evidence
framework and is not yet connected to training.
