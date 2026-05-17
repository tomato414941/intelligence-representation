# Shogi Learning Boundaries

This document records shogi-specific learning boundaries. It does not define a
generic world/problem framework for every future domain.

## World-Side Shogi Data

`intrep.worlds.shogi` owns source-side shogi data and formats:

- `ShogiGameRecord`: lightweight recorded game facts such as actors, initial
  position, moves, result, end reason, and source metadata
- `ShogiGameTrace`: derived replay expansion such as positions, legal moves,
  side-to-move, next positions, rewards, and done flags
- KIF / USI parsing and writing
- legal shogi game records and source-derived game-record JSONL
- Experience Store
- Training Data Bundle creation
- train/eval game-record splitting
- engine-analysis source records

Code belongs in `worlds/shogi` when it preserves, organizes, validates,
selects, or replays shogi experience before a specific learning target is
chosen.

## Policy/Value Problem Data

`intrep.problems.shogi_policy_value` owns the shogi policy/value learning
problem:

- Data Selection loading for policy/value training
- conversion from selected game traces to `ShogiPolicyValueExample`
- tensorized policy/value samples and tensor caches
- policy/value model training and evaluation
- generated-data training cycles
- Online Replay orchestration

Code belongs in `problems/shogi_policy_value` when it depends on the
policy/value target, model input tensors, loss, metrics, checkpoint training
configuration, or learner loop.

## Boundary Rule

The boundary is source-side versus problem-side.

Training Data Bundles are world-side fixed source snapshots. Tensor caches are
problem-side acceleration artifacts derived from a Data Selection or Training
Data Bundle.

`ShogiGameRecord` is not a cache. It should not store replay-derived legal
moves, next positions, rewards, or problem-derived targets. Those belong in
`ShogiGameTrace` when computed in memory, or in a rebuildable cache when a run
needs to avoid recomputing them.

Experience Store and Online Replay Buffer are independent. Experience Store is
durable source storage for generated or collected shogi experience. Replay
Buffer is dynamic learner state used by Online Replay.

## Online Replay Eval

Online Replay uses a fixed `training_eval_data_selection` for training-time
evaluation, early stopping, and best-checkpoint selection.

Generated games are learner experience. They are added to the replay buffer and
are not split into a generated eval holdout. Playing-strength evaluation is a
separate post-training concern.

## Online Replay Generation Lifecycle

Online Replay can run multiple iterations. The checkpoint used to generate
experience should be treated as an experience generator, not merely as a base
checkpoint.

A newly trained candidate checkpoint must be evaluated before it is used to
generate more replay experience. If the candidate loses that evaluation, the
learning workflow should stop at that candidate instead of generating more
experience from the losing checkpoint.

## Playing Strength Evaluation

Playing-strength evaluation is a player-vs-player match. The project-facing
entry point uses `player_a` and `player_b`; it does not name one side as the
opponent. Side assignment is handled by the arena evaluator and should be
recorded with the match result.

The durable evidence is the game-record JSONL. The stdout match summary is a
derived convenience view, not a second source of truth.

## Evaluation Boundaries

Training-time eval, playing-strength eval, inference-performance eval, and
learning experiment summaries are separate roles:

- training metrics are the source for loss, early stopping, and best-checkpoint
  selection
- player match game records are the source for playing-strength evidence
- inference-performance docs summarize latency, throughput, and CPU/GPU behavior
- learning experiment docs summarize conclusions and cite evidence without
  duplicating raw artifacts

## Online Replay Progress

Policy/value training reports generic training progress events. Online Replay
owns the iteration context for those events and prints lightweight progress lines
that include the iteration, step, loss, elapsed time, replay size, sampled examples,
and fixed training-eval example count.

RunPod Online Replay jobs set `PROGRESS_EVERY=100` unless overridden.

## Online Replay Step Budget

Online Replay owns the per-iteration training budget. A run specifies
`sampled_examples_per_iteration`, `training_batch_size`, `target_sample_passes`,
and an optional `max_optimizer_steps_per_iteration`.

The policy/value training loop still runs on optimizer steps, but Online Replay
derives those steps from the budget before calling the training loop.

Iteration metrics record `effective_sample_passes` as:

```text
actual_steps * training_batch_size / sampled_examples
```

This is an accounting metric. The intended budget is
`target_sample_passes`; the effective value records what actually ran after
step capping or early stopping.

Do not introduce a shared world/problem data framework from shogi alone.
