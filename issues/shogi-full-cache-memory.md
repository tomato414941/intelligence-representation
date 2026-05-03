# Shogi Full-Cache Memory Stability

Status: open.

## Issue

The Qhapaq shogi move-choice full cache is currently loaded as a large Python
object list. This is operationally fragile for long RunPod jobs, especially
when PyTorch `DataLoader` workers are enabled.

This is separate from raw DataLoader throughput. The immediate throughput
control path is covered by
[`closed/shogi-dataloader-throughput.md`](closed/shogi-dataloader-throughput.md).
This issue tracks memory stability and cache format.

## Evidence

| Observation | Result |
| --- | --- |
| full train JSONL load, local measurement | 2,220,818 examples; max RSS about 12.5 GB |
| 50-step full-cache smoke, RunPod EU-RO-1, `num_workers=4` | passed |
| 2000-step full-cache baseline, RunPod US-CA-2, `num_workers=4` | reached 350/2000 steps, then SSH timed out with pod not responding |
| 2000-step full-cache baseline, RunPod EU-RO-1, `num_workers=0` | passed |
| local 100k-example DataLoader probe, `num_workers=4` | worker processes increased proportional set size as they touched the object cache |

The failed 2000-step `num_workers=4` run did not show CUDA memory pressure:
CUDA max memory was about 8.1 GB before the pod stopped responding. The more
likely risk is CPU RAM pressure from Python object cache sharing breaking down
across worker processes.

## Current Mitigation

The RunPod shogi training script defaults to:

| Setting | Default | Reason |
| --- | --- | --- |
| `NUM_WORKERS` | `0` | Avoid private-copy RAM growth from DataLoader workers. |
| `DATA_CENTER_IDS` | unset | Leave scheduling flexible by default; use `EU-RO-1` for longer baselines because it has a successful full-cache run. |

For longer full-cache RunPod baselines, prefer:

```sh
DATA_CENTER_IDS=EU-RO-1
NUM_WORKERS=0
```

## Candidate Direction

Move away from a full Python-object JSONL dataset for training.

Candidate paths:

| Direction | Purpose |
| --- | --- |
| tensorized cache | Store position and move features in tensors or arrays that workers can share or memory-map safely. |
| binary / memory-mapped cache | Avoid reparsing JSON and reduce Python object pressure. |
| streaming dataset | Avoid loading the whole cache into RAM, with careful shuffle strategy. |
| periodic checkpoint / metrics | Preserve partial results if a pod becomes unreachable during a longer run. |

## Acceptance Criteria

This issue can close when at least one of these is true:

- full-cache training can safely use DataLoader workers without large CPU RAM
  growth, or
- worker-free full-cache training is measured as acceptable for the intended
  baseline scale, and the operational default is documented as intentional.

Before increasing `NUM_WORKERS` above zero for full-cache runs, measure CPU RAM
behavior on the target cache and pod size.
