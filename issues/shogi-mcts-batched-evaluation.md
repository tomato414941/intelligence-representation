# Shogi MCTS Batched Evaluation

Status: open. Priority: medium.

## Issue

Current checkpoint-vs-checkpoint MCTS evaluation can run on CUDA, but it likely
does not use the GPU efficiently. MCTS evaluates many small positions while the
search loop, legal move generation, and tree bookkeeping stay on CPU. Sending
one position at a time to the model can leave the GPU underutilized.

This matters for external one-game play as well as batch evaluation. Running
many games in parallel can improve evaluation throughput, but it does not solve
the single-game case.

For Floodgate-like play, the target metric is wall-clock move latency and search
quality under the time limit, not raw GPU throughput. Batch waiting must stay
small enough that it does not make the engine miss useful thinking time.

## Desired Direction

Add a single-game MCTS path that can collect multiple leaf positions and evaluate
them as one model batch.

The first version should stay shogi-specific and small:

- keep the current MCTS path as the reference behavior
- batch only neural-network position evaluation
- avoid changing game rules, move legality, or checkpoint loading
- record GPU utilization or latency before claiming an efficiency improvement

## Risks

- Pending leaf nodes can be selected repeatedly without a virtual-loss or
  similar mechanism.
- Waiting too long to fill a batch can hurt move latency in external play.
- Batched evaluation can make MCTS behavior harder to test if tree updates and
  model calls are mixed together.

## Acceptance Criteria

- one-game MCTS can evaluate more than one leaf position per model call
- behavior is covered by a small deterministic test or smoke evaluation
- CUDA usage is measured with at least a basic `nvidia-smi` or timing log
- the implementation does not require running multiple games in parallel

## Progress

2026-05-09:

- `shogi-arena-agent` added batch evaluation support to
  `ShogiMoveChoiceCheckpointEvaluator`.
- CPU smoke confirmed that batch evaluation matches repeated single evaluation
  for the promoted d256-h1024-heads8-l6-shogi checkpoint.
- MCTS does not yet use batched leaf evaluation, so this issue remains open.
