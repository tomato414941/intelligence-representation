# Online Replay Buffer Persistence

Status: closed. Priority: low.

## Issue

Online Replay currently keeps its `ReplayBuffer` in process memory. When a run
ends or is interrupted, the buffer contents are discarded. A resumed run starts
from an empty buffer.

This is acceptable for short smoke tests and early Online Replay experiments,
but it can become costly for longer runs where the accumulated replay buffer is
part of the experiment state.

## Scope

This issue is about persisting and resuming the learner-facing Online Replay
Buffer state.

It is independent of Experience Store. Experience Store is not the input,
output, persistence layer, or source of truth for Online Replay Buffer state.

## Possible State

A future implementation may need to persist:

- buffered `ShogiMovePolicyValueExample` items
- replay capacity
- sampling seed or random state
- completed cycle index
- checkpoint identity used for the next cycle
- actor/search settings that produced buffered examples

## Non-Goals

- Do not introduce this before Online Replay needs long-running resumable runs.
- Do not couple replay-buffer persistence to Experience Store.
- Do not change offline fixed-data training CLIs.
- Do not add prioritized replay, target networks, or distributed replay as part
  of this issue.

## Acceptance Criteria

- Online Replay can resume from a saved replay-buffer state.
- The saved state records enough metadata to avoid accidentally resuming with
  incompatible replay capacity, checkpoint, or actor/search settings.
- Short smoke runs can still run without persistence.
- Experience Store remains independent from Online Replay Buffer persistence.

## Resolution

Online Replay now supports `resume=True` / `--resume`.

The implementation does not serialize the generic `ReplayBuffer`. Instead, it
reconstructs generated replay state from completed iteration artifacts in the
run directory:

- per-source `generated-games.jsonl`
- `generation-summary.json`
- iteration `metrics.json`
- previous iteration checkpoints

Resume validates the replay capacity, seed Data Selection, initial checkpoint
identity, MCTS settings, max plies, and generated experience source metadata
before continuing. It restores the next checkpoint from the last completed
iteration according to the configured `next_checkpoint` policy, rebuilds the
generated replay samples, and starts at the first incomplete iteration.

Experience Store remains independent. It may durably store generated game
records, but it is not the source of truth for Online Replay Buffer state.

## Related

- [`online-experience-replay-orchestration.md`](closed/online-experience-replay-orchestration.md)
- [`mixed-source-store-continual-learning.md`](../mixed-source-store-continual-learning.md)
