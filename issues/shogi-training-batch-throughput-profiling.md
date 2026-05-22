# Shogi Training Batch Throughput Profiling

Status: open. Priority: low.

## Issue

The minimal split-global shogi policy/value training throughput is not
monotonic with batch size on RunPod RTX 3090 measurements.

Measured 500-step runs:

- batch 512: about 5.15 steps/sec, about 2636 examples/sec
- batch 1024: about 1.66 steps/sec, about 1699 examples/sec

Batch 1024 fits in GPU memory, and DataLoader wait time is small, but
forward/backward time grows more than expected. The current evidence supports
using batch 512 for full training, but it does not fully explain why batch
1024 is slower.

## Desired Shape

Treat this as a throughput profiling issue, not a training blocker.

If revisited, isolate the cause with controlled measurements:

- compare batch 512 and 1024 on the same pod
- optionally include batch 768
- keep GPU type, vCPU/RAM, data center, model config, and cache constant
- profile a short fixed-step window with `torch.profiler`
- separate cache restore/setup time from training-loop time
- inspect whether action-plane output/loss, Transformer blocks, memory
  bandwidth, or CPU/GPU synchronization dominates

## Current Policy

Use batch 512 for full training unless a later controlled run shows a better
examples/sec setting.

## Non-Goals

- delay full training for this investigation
- optimize cache restore networking here
- change model architecture based only on these throughput runs
- treat 500-step eval loss as the primary comparison metric

## Evidence

See `docs/shogi/training-throughput.md` for the resource-monitored batch 512
and batch 1024 measurements.
