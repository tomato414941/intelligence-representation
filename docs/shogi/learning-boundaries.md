# Shogi Learning Boundaries

This document records shogi-specific learning boundaries. It does not define a
generic world/problem framework for every future domain.

## Domain-Side Shogi Data

`intrep.domains.shogi` owns source-side shogi data and formats:

- `ShogiGameRecord`: lightweight recorded game facts such as actors, initial
  position, moves, result, end reason, and source metadata
- `ShogiGameTrace`: derived replay expansion such as positions, legal moves,
  side-to-move, next positions, rewards, and done flags
- KIF / USI parsing and writing
- legal shogi game records and source-derived game-record JSONL
- Training Data Bundle creation
- train/eval game-record splitting
- engine-analysis source records

Code belongs in `domains/shogi` when it preserves, organizes, validates,
selects, or replays shogi experience before a specific learning target is
chosen.

## Policy/Value Problem Data

`intrep.problems.shogi_policy_value` owns the shogi policy/value learning
problem:

- Data Selection loading for policy/value training
- conversion from selected game traces to durable `ShogiMovePolicyValueExample` records
- tensorized policy/value samples and tensor caches
- policy/value model training and evaluation
- Online Replay orchestration

Code belongs in `problems/shogi_policy_value` when it depends on the
policy/value target, model input tensors, loss, metrics, checkpoint training
configuration, or learner loop.

## Boundary Rule

The boundary is domain-side versus problem-side.

Training Data Bundles are domain-side fixed source snapshots. Tensor caches are
problem-side acceleration artifacts derived from a Data Selection or Training
Data Bundle.

`ShogiGameRecord` is not a cache. It should not store replay-derived legal
moves, next positions, rewards, or problem-derived targets. Those belong in
`ShogiGameTrace` when computed in memory, or in a rebuildable cache when a run
needs to avoid recomputing them.

Online Replay resume reconstructs generated replay state from completed
iteration artifacts in the run directory. It does not serialize the generic
`ReplayBuffer` object.

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
generate more replay experience. This gate is a degradation guard, not proof
that the candidate is stronger. Clearly worse candidates stop the loop; close
results may continue but should be interpreted as unclear rather than as
strength improvement.

Gate results record the candidate's side-specific result split for
interpretation. Side skew is not currently a stop condition; it is preserved so
future experiment summaries can say whether side bias was present.

Generation uses game-level worker processes for produced experience, and the
generator gate uses match-level worker processes for checkpoint-vs-checkpoint
evaluation. `NN leaf eval batch limit` is the neural evaluator batch cap used by
both single-game in-tree leaf batching and generated-game multi-position
batching.

## Online Replay Generated Data Quality

Games that reach the `max_plies` cap are valid generated experience, but they
are generation-quality evidence. They are kept in the generated game records and
converted into policy/value examples. With winner-based value targets, a
`max_plies` draw has no winner, so its value target is unknown and the value
loss mask excludes it; its policy target remains usable.

Generation summaries record `max_plies_draw_count`, `max_plies_draw_rate`,
`game_over_count`, and `game_over_rate` separately from the raw `end_reasons`.
Online Replay does not filter or downweight cap-draw records by default. A high
cap-draw rate should be interpreted in run summaries as a generator-quality
signal, not as an invalid source-record condition.

## Online Replay Gate Cost

The generator gate is a training-control cost, not final playing-strength
evaluation. Its job is to stop clearly worse candidate generators before they
produce more replay experience.

The default gate is intentionally smaller than a full strength evaluation. Gate
settings are part of the Online Replay configuration: gate games, gate worker
processes, MCTS simulations, NN leaf eval batch limit, and max plies. Iteration
metrics record those settings, gate wall time, and the gate result so run
summaries can judge whether the guard cost was worth paying.

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
