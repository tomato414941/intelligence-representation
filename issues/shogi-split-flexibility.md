# Shogi Split Flexibility

## Issue

The current Qhapaq move-choice cache is split before example generation, but the
generated examples do not carry a game identifier. This is safe for the current
fixed train/eval split, but the example cache cannot be safely re-split later.

## Current State

| Area | Current behavior |
| --- | --- |
| source records | `qhapaq_all_games.jsonl` has one game per line |
| split | train/eval files are produced at game boundaries |
| generated examples | examples contain position, legal moves, chosen move, and optional value target |
| missing metadata | generated examples do not include `game_index` or `ply_index` |

## Problem

If a future run needs a different split, k-fold evaluation, or a shuffled
game-level split, the existing example cache is not enough. Splitting generated
examples by row would leak correlated positions from the same game across
train/eval.

## Candidate Direction

Keep the current fixed split for near-term experiments. If split flexibility
becomes important, add source metadata such as `game_index` and `ply_index` to
the shogi move-choice example cache, then split by game id rather than row.
