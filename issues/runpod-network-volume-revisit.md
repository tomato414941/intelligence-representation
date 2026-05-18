# RunPod Network Volume Revisit

Status: open. Priority: low. Verification needed.

## Issue

Current shogi RunPod jobs use disposable container disk plus input/output sync:

```text
--container-disk-size 80
--volume-size 0
```

This may already be sufficient for current shogi training and evaluation runs,
where the required inputs and outputs appear practical to sync for each Pod.

RunPod network volumes may become useful when dataset/cache reuse, checkpoint
durability, or repeated sweeps make repeated sync too expensive. They also add
startup, mount, and I/O behavior that should be measured before becoming the
default.

## Current Position

Keep network volumes disabled for the current shogi RunPod script.

The existing evidence is not strong enough to claim that network volumes caused
specific readiness failures. Treat the current `--volume-size 0` default as a
KISS operational choice, not as a proven diagnosis.

The original concern may be partially resolved by the current Training Example
JSONL flow. Older tensor-cache runs recorded about 100 seconds of repo/cache
sync. The 2026-05-18 RunPod smoke with `qhapaq-full` Training Example JSONL
recorded about 11 seconds of repo/input sync for a 645 second job. That single
measurement suggests sync is not currently the dominant cost, but it should be
confirmed across the next real training runs before closing this issue.

## Revisit Triggers

Revisit network volumes when at least one of these becomes true:

- input sync time is a meaningful share of total run time
- tensorized shogi cache artifacts become large and reused across many runs
- checkpoint durability during long runs matters more than disposable simplicity
- repeated hyperparameter sweeps resend the same large inputs
- RunPod startup/readiness failures need a controlled volume/no-volume comparison

Also close or downgrade this issue if several current Training Example JSONL
runs show that input sync remains a small share of total wall time.

## Evaluation Plan

Compare network-volume and no-network-volume runs with the same image, GPU type,
data center, and workload:

- Pod creation and SSH readiness time
- setup time
- input availability time
- training/evaluation wall time
- output sync time
- failure mode, if any
- cost impact

Keep the first comparison small. Do not switch the default until the result is
better than the current container-disk flow for a concrete workload.

## Acceptance Criteria

This issue can close when either:

- a measured workload justifies network volumes and the RunPod script is updated
  intentionally, or
- network volumes are explicitly rejected for the current project phase with
  evidence that container disk plus sync remains simpler and sufficient.
