# Shogi Generated Data MCTS Visit Target Adoption

Status: open
Priority: high

## Problem

`intelligence-representation` can construct shogi policy targets from MCTS root
child visit counts, and Online Replay already uses `mcts_visit_counts` for
generated games.

The generated-data training cycle still constructs generated examples with
`policy_target_construction="chosen_move"`. That can discard most of the MCTS
search signal when generated checkpoint games are used as supervised training
data.

## Desired Shape

Generated checkpoint MCTS records should use MCTS visit-count policy targets
when the records contain the required search evidence.

The boundary should stay explicit:

- source records preserve raw MCTS search evidence
- target construction chooses `mcts_visit_counts`
- training examples receive normalized policy targets

Do not make non-MCTS sources pretend to have visit-count targets.

## Close Condition

- The generated-data training path can request `mcts_visit_counts` instead of
  `chosen_move` for checkpoint MCTS records.
- Tests cover generated-game target construction using visit counts.
- Missing visit-count evidence fails clearly or falls back only when explicitly
  configured.
