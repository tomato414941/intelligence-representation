# Shogi Play Evaluation Runtime Efficiency

## Problem

Short shogi playing-strength checks spend most of their wall time outside the
actual match.

The 2026-05-23 Suisho5 smoke check used one game with `MCTS_SIMULATIONS=1` and
`go nodes 1`. The game itself finished quickly, but the RunPod job took about
292 seconds end to end.

Observed costs:

- `repo_sync`: about 144 seconds
- `setup`: about 51 seconds
- `remote_1`: about 50 seconds, including YaneuraOu build, NNUE setup, and the
  actual match
- match progress elapsed time: about 1.3 seconds for the single game

This makes small evaluation loops expensive and obscures the real cost of model
inference or game play.

## Desired Shape

Playing-strength evaluation should separate reusable environment preparation
from the measured match.

The evaluation path should avoid repeatedly doing work that is independent of
the tested model entry:

- avoid syncing large unchanged checkpoint artifacts when possible
- avoid rebuilding YaneuraOu for every small check
- avoid downloading and extracting the same NNUE archive repeatedly
- make setup time and match time visibly separate in run artifacts

The goal is not just lower cost. The goal is to make evaluation results easier
to interpret: when a check is slow, it should be clear whether the time went to
environment setup, engine setup, MCTS, model inference, or opponent response.

## Non-Goals

- Do not add a large evaluation framework just for this issue.
- Do not hide setup costs by silently reusing stale state.
- Do not require this before running an occasional one-off check.

