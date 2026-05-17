# Shogi Player Match Progress Visibility

Status: closed
Priority: medium

## Problem

Long shogi player matches can remain silent until the whole match finishes.

During the 2026-05-17 RunPod check against YaneuraOu MaterialLv1 `go nodes
1000`, a 100-game match ran for about 18 minutes in the remote evaluation step.
The process was alive, but stdout did not show completed-game progress, the
summary file stayed empty until the end, and normal status required a second SSH
session to inspect processes and GPU usage.

This makes it hard to distinguish a healthy long match from a stalled engine,
stuck game, dead CUDA worker, or output buffering issue.

## Desired Shape

Player-match evaluation should emit coarse progress during long matches without
making short checks noisy.

At minimum:

- print completed games out of total games
- print elapsed seconds
- print current win/loss/draw counts
- flush progress output
- keep the mechanism in the player-match evaluation path, not in RunPod wrapper
  scripts

## Close Conditions

- A 100-game player match emits periodic progress before the final summary.
- Progress can confirm the match is alive without opening a second SSH session.
- The final summary remains the durable result format.

## Resolution

`shogi-arena-agent` now supports `--progress-every-games` in
`scripts/evaluate_shogi_players.py`. The RunPod player-match wrapper sets
`PROGRESS_EVERY_GAMES=10` unless overridden, so long RunPod matches emit sparse
completed-game progress before the final JSON summary.
