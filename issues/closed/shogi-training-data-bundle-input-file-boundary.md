# Shogi Training Data Bundle Input File Boundary

Status: closed.

## Issue

Shogi Training Data Bundle creation can accept multiple training game-record files, but
unbounded multi-file inputs can make source file management harder.

Multiple inputs are useful because they avoid creating ad-hoc combined JSONL
files. However, if normal operation becomes a long list of run outputs and
temporary record files, the project can lose track of what the training data
actually is.

## Risk

- run outputs and durable record sets may be mixed in one command
- similar record files may be passed repeatedly or accidentally duplicated
- command lines may become the only place where the dataset composition exists
- manifests may list many files but still be hard to reason about
- source file count may grow instead of converging toward stable record sets

## Current Policy

Treat multiple `--train-games` inputs as an escape hatch, not the default data
management model.

The default path should be:

```text
many generated files
  -> append / consolidate into a stable record set
  -> create Training Data Bundle from that stable record set
```

Training Data Bundle manifests should still record all input files when multiple files
are used.

## Acceptance Criteria

- Decide whether `create_shogi_training_data_bundle.py` should accept multiple
  training game-record files.
- If multiple inputs remain supported, document that stable record sets are the
  preferred normal input.
- Avoid requiring users to create ad-hoc combined JSONL files only to build a
  Training Data Bundle.

## Non-Goals

- redesign Experience Store
- introduce a generic dataset registry
- remove support for temporary experiment inputs

## Resolution

Keep repeated `--train-games` support. It is useful for temporary experiments
and explicit source mixes, and avoids requiring ad-hoc combined JSONL files.

The normal durable path is still a stable record set or Experience
Store-derived game-record JSONL. `scripts/create_shogi_training_data_bundle.py`
now warns when multiple train inputs are passed, and bundle manifests continue
to record every original train source path.
