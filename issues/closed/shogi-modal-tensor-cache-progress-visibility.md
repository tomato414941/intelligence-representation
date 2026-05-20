# Shogi Modal Tensor Cache Progress Visibility

Status: closed. Priority: medium.

## Issue

The Modal shogi tensor-cache builder can run for a long time without emitting
local progress after startup. During a full `qhapaq-full` policy-plane cache
build, the local runner only showed the initial Modal object creation and then
stayed silent while remote shard tasks were running.

This makes it hard to distinguish:

- healthy long-running shard construction
- stalled remote work
- slow upload or release
- a worker that will later fail with OOM or another resource error

## Desired Shape

Long-running Modal cache builds should report coarse, useful progress without
turning the build script into an operations dashboard.

Useful signals:

- remote bundle upload complete
- remote cache reset complete
- total shard task count
- periodic completed shard count
- split/source/index range for completed or failed shards
- manifest write start and finish
- local release start and finish

The output should remain line-oriented and easy to read in a terminal.

## Non-Goals

- add a database or service for job tracking
- build a full Modal dashboard wrapper
- make cache construction depend on external observability infrastructure
- change the tensor cache format

## Resolution

The Modal tensor-cache builder now emits line-oriented JSON events for upload,
reset, build start, periodic shard progress, manifest writing, local release,
and final completion.

Shard progress is based on committed remote shard manifest files under the
Modal Volume cache directory. This makes progress reporting independent of
worker stdout and compatible with shard resume after worker restart or
preemption.
