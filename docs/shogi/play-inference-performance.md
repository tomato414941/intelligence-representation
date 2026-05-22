# Shogi Play Inference Performance

This document records measured latency for shogi one-game move decisions. It is
not a run log, a model-quality report, or a cloud cost ledger.

`runs/` is disposable. Measurements that should survive must be summarized here
or in a promoted model note.

## Current Reading

The most relevant current Floodgate-like latency measurements are the
2026-05-13 YaneuraOu one-game checks below.

## Required Context

Record enough context to explain latency and throughput:

- inference path
- model identity
- environment
- input shape
- output shape
- request definition
- output unit definition
- runtime settings
- workload used for measurement

## Required Metrics

- `request_wall_time_sec`: end-to-end wall-clock time for one request.
- `model_call_count`: number of model calls during one request.
- `model_wall_time_sec`: total wall-clock time spent in model calls.
- `non_model_wall_time_sec`: wall-clock time outside model calls.
- `output_count`: number of output units produced or evaluated.
- `output_per_sec`: `output_count / request_wall_time_sec`.

## Measurement Conditions

Unless noted otherwise:

- Entry point: `evaluate_shogi_players.py`
- Model: d256-h1024-heads8-l6-shogi
- Workload: checkpoint vs YaneuraOu MaterialLv1 `go nodes 1`
- Request: one move decision
- Output unit: MCTS simulation
- Board backend: `cshogi`
- Max plies: 320

### Detailed Measurements

| Case | Date | Model | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Runtime image | MCTS simulations per move | NN leaf eval batch limit | Actual NN leaf eval batch avg | Actual NN leaf eval batch max | Move time limit | Games | Avg request wall | P95 request wall | Max request wall | Avg model calls | Avg model wall | Avg non-model wall | Avg output/sec | Sample count | Container CPU util avg | Container CPU util max | GPU util avg | GPU util max | GPU memory used | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `rtx4000ada_mcts1024_b64` | 2026-05-13 | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 5 vCPU, 47 GB | not recorded | not recorded | $0.20/hr | not recorded | 1024 | 64 | not measured | not measured | 9.0s | 1 | 1.625s | 3.619s | 4.862s | 27.906 | 1.324s | 0.301s | 764.8 | 53 | 19.3% | 30.6% | 8.3% | 16.0% | 1602 MiB | 0-1, game_over, 64 plies | Stayed below 10s with margin; low GPU utilization means the measured model wall time does not imply GPU saturation. |
| `rtx4000ada_mcts2048_b64` | 2026-05-13 | d256-h1024-heads8-l6-shogi | RTX 4000 Ada | 5 vCPU, 47 GB | not recorded | not recorded | $0.20/hr | not recorded | 2048 | 64 | not measured | not measured | 9.0s | 1 | 2.944s | 3.995s | 6.715s | 45.192 | 2.421s | 0.523s | 742.1 | 76 | 18.4% | 32.2% | 8.9% | 16.0% | 1138 MiB | 0-1, game_over, 52 plies | Stayed below 10s in this one-game check; GPU utilization remained low, so increasing simulations mostly increases serialized work rather than saturating the GPU. |
| `rtxa4000_mcts1024_b64_actualbatch` | 2026-05-13 | d256-h1024-heads8-l6-shogi | RTX A4000 | 14 vCPU, 62 GB | not recorded | not recorded | $0.17/hr | not recorded | 1024 | 64 | 51.3 | 64 | 9.0s | 1 | 1.312s | 1.595s | 2.231s | 22.125 | 1.053s | 0.259s | 824.8 | 64 | 2.0% | 5.0% | 15.0% | 27.0% | 1537 MiB | 0-1, game_over, 96 plies | Actual leaf batches often approached the configured limit, but GPU utilization stayed low. |
| `rtxa4000_mcts2048_b64_actualbatch` | 2026-05-13 | d256-h1024-heads8-l6-shogi | RTX A4000 | 14 vCPU, 62 GB | not recorded | not recorded | $0.17/hr | not recorded | 2048 | 64 | 42.6 | 64 | 9.0s | 1 | 2.329s | 4.155s | 4.255s | 54.286 | 1.877s | 0.452s | 992.7 | 66 | 2.1% | 5.4% | 16.2% | 31.0% | 757 MiB | 0-1, game_over, 56 plies | Actual leaf batches still reached 64, but average batch size fell versus MCTS1024 and GPU utilization remained low. |

## Notes

- Container CPU and GPU utilization columns are populated from
  `gpu_summary.json` produced by the RunPod evaluation wrapper. Sampling
  interval was 1 second.
- Game-level worker processes are not listed because this workload is one live
  game; sharding other games would not speed up the current move.
- Concurrent games per process is not listed because there is only one current
  game.
- The relevant batching mechanism is one-tree MCTS leaf batching through
  `NN leaf eval batch limit`.
- `Actual NN leaf eval batch` measures how many leaf positions were actually
  sent to one model call. It excludes the root expansion call.
