# Shogi Checkpoint Immutable Identity

Status: closed
Priority: high

## Problem

Shogi experiment records currently have to refer to checkpoints by path, such as
`models/d256-h1024-heads8-l6-shogi/checkpoint.pt`.

That path is not an immutable identity. If a trained checkpoint is later
promoted into the same model path, the path remains the same while the checkpoint
contents change. Historical experiment records can then become ambiguous:

- the path used at run time
- the checkpoint contents used at run time
- the current checkpoint contents at that path

These are different concepts and should not be collapsed into one field.

## Desired Shape

Checkpoints that are used as durable inputs or outputs for shogi experiments
should have an immutable identity independent of their mutable filesystem path.

Experiment records should be able to store:

- checkpoint identity
- path used at run time
- model architecture/config
- training provenance or source run when available
- checksum or equivalent content fingerprint when practical

The mutable `models/.../checkpoint.pt` path can remain a convenient current
adoption path, but it should not be the only way to identify a checkpoint in
experiment records.

## Close Condition

- Shogi checkpoint artifacts expose or can be assigned an immutable identity.
- Learning experiment records include checkpoint identity separately from
  run-time path.
- Promotion into `models/.../checkpoint.pt` does not make old experiment records
  ambiguous.

## Resolution

2026-05-20:

- Shogi policy/value checkpoints now store a canonical content identity in the
  checkpoint config:
  - `checkpoint_id`
  - `checkpoint_sha256`
- The identity is computed from checkpoint schema, model/input config, and
  model state dict contents, not from the mutable filesystem path.
- Checkpoint load paths validate that the stored identity still matches the
  checkpoint contents.
- Game generation and online replay gate commands now pass the immutable
  checkpoint identity as `--*-checkpoint-id`, while preserving the run-time
  path as `--*-checkpoint`.
- Training metrics and online replay metrics now record checkpoint identity
  separately from checkpoint path.
- Experience-store checkpoint actor counts now prefer `checkpoint_id` over
  `checkpoint` path when grouping checkpoint actors.
