# Shogi Training Throughput

This document records measured throughput for shogi policy/value training. It
is not a run log, a model-quality report, or a cloud cost ledger.

`runs/` is disposable. Measurements that should survive must be summarized here
instead of relying on run-local paths.

This document is about learning runtime behavior. Cost estimates belong in
`docs/compute-costs.md`, generated game throughput belongs in
`docs/shogi/self-play-generation-throughput.md`, and play-time inference
latency belongs in `docs/shogi/play-inference-performance.md`.

## Current Reading

The current full RunPod training run is still in progress. It is using the
minimal split-global shogi position input with action-plane policy/value output
on an RTX 4000 Ada GPU. The latest observed training speed is about 3.09
steps/sec with high GPU utilization and low data wait.

## Required Context

Record enough context to explain training throughput:

- training entrypoint
- model entry
- input artifact
- output target representation
- hardware
- batch size
- optimizer step count
- eval and checkpoint cadence
- cache restore behavior
- DataLoader worker count
- training-loop runtime
- end-to-end runtime

## Required Metrics

- `steps_per_second`: optimizer steps per second during the training loop.
- `examples_per_second`: `steps_per_second * batch_size`.
- `data_wait_seconds`: time spent waiting for the next batch.
- `forward_backward_seconds`: model forward, loss, backward time.
- `optimizer_seconds`: optimizer and scheduler update time.
- `eval_seconds`: wall-clock time for one periodic eval.
- `gpu_util`: observed GPU utilization during training.
- `gpu_memory_used`: observed GPU memory during training.
- `training_loop_runtime`: runtime for optimizer-loop work.
- `end_to_end_runtime`: setup, cache restore, training, output sync, and
  cleanup runtime.

## Purpose

Measure optimizer-loop throughput and resource use. This is different from
self-play generation throughput, where search and game orchestration dominate,
and from play-time inference latency, where one-move response time dominates.

## Measurement Conditions

Unless noted otherwise:

- Entry point: `train_shogi_policy_value.py`
- Workload: fixed shogi policy/value training from a tensor cache
- Device: cuda
- Batch size: record the actual configured training batch size
- Eval cadence: record the configured periodic eval interval
- Checkpoint cadence: record the configured checkpoint interval
- RunPod template / torch version: record the actual values used by each
  measurement.

## Detailed Measurements

| Case | Date | Status | Model entry | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Input artifact | Output target | Batch | Steps | Eval cadence | Checkpoint cadence | DataLoader workers | Cache restore size | Cache restore time | Training loop runtime | End-to-end runtime | Steps/sec | Examples/sec | Data wait | Forward/backward | Optimizer | Eval time | GPU util | GPU memory used | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `minimal_split_global_action_plane_rtx4000ada_100k` | 2026-05-22 | running | `shogi-policy-plane-minimal-split-global` | RTX 4000 Ada | not recorded | not recorded | not recorded | $0.20/hr | Qhapaq full packed tensor cache | action-plane policy/value | 512 | 100000 | 1000 steps | 1000 steps | 2 | about 2.0 GB | not recorded | running | running | about 3.09 | about 1582 | about 0.03-0.10s / 100 steps | about 11.3-11.8s / 100 steps | about 0.27-0.32s / 100 steps | about 12.7-13.0s | observed 98% | observed 7150 MiB / 20475 MiB | Running measurement; finalize after completion. |

## Notes

- Timing columns should be taken from training progress logs when available.
- GPU utilization and memory should be sampled from the same pod while training
  is active.
- Keep quality interpretation out of this document; this page is for runtime
  behavior.
