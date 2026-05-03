# RunPod Region Stability

Status: open.

## Issue

Shogi training runs have shown different stability across RunPod regions. This
is operational evidence for choosing a region before starting longer runs, not a
compute-cost or model-quality record.

## Evidence

| Observation | Result |
| --- | --- |
| 2000-step shogi full-cache baseline, RunPod US-CA-2 | reached 350/2000 steps, then the pod stopped responding over SSH |
| 2000-step shogi full-cache baseline, RunPod EU-RO-1 | completed |

## Current Mitigation

Prefer `DATA_CENTER_IDS=EU-RO-1` for longer shogi training baselines until more
region evidence exists.

## Related Issues

- [`shogi-full-cache-memory.md`](shogi-full-cache-memory.md) tracks Python
  object-list memory stability for the full cache.
- [`closed/shogi-dataloader-throughput.md`](closed/shogi-dataloader-throughput.md)
  tracks the closed DataLoader throughput-control issue.

## Acceptance Criteria

This issue can close when either:

- EU-RO-1 remains the documented preferred region after enough successful long
  shogi runs, or
- region choice is no longer material because multiple regions complete
  comparable shogi runs reliably.
