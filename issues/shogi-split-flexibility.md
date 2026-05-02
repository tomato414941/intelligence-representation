# Shogi Split Flexibility

Status: resolved for the current cache format.

## Issue

The Qhapaq move-choice cache must support game-level train/eval splits without
leaking positions from the same game across splits.

## Current State

| Area | Current behavior |
| --- | --- |
| source records | `qhapaq_all_games.jsonl` has one game per line |
| split | train/eval files are produced at game boundaries |
| generated examples | examples contain position, legal moves, chosen move, optional value target, `game_index`, and `ply_index` |
| cache | train/eval caches were regenerated with game and ply metadata |

## Problem

Splitting generated examples by row would leak correlated positions from the
same game across train/eval. The current cache avoids this by splitting source
games before example generation and retaining source game metadata in each
example.

## Candidate Direction

Keep the current fixed split for near-term experiments. If k-fold or shuffled
game-level splits become important, use `game_index` metadata rather than row
numbers.
