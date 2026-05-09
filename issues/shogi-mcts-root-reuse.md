# Shogi MCTS Root Reuse

Status: open. Priority: medium.

## Issue

Current shogi MCTS play can discard search work after each move. In external
one-game play, especially Floodgate-like play, the next position is often a
direct child of the previous root. Reusing that subtree can save wall-clock time
without increasing model size or GPU cost.

This is separate from batched leaf evaluation. Batching makes model calls more
efficient; root reuse avoids repeating already-computed search.

## Desired Direction

Keep the first version shogi-specific and small:

- after the engine plays a move, keep the matching child as the new root
- after the opponent plays a move, keep the matching child if it exists
- discard the tree when the move is not found or the position is inconsistent
- do not change move legality or model inference behavior

## Risks

- Reusing stale tree statistics can hurt play if position identity or turn
  handling is wrong.
- Draw, repetition, or special terminal states may make reuse bugs subtle.
- This should not be mixed with ponder support until root reuse is reliable.

## Acceptance Criteria

- repeated play can carry the MCTS tree across at least one legal move
- reuse falls back cleanly when the played move is not in the tree
- a small deterministic test covers both reuse and fallback
