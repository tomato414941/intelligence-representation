# Inference Performance

This document catalogs wall-clock inference and interaction performance. It is
not a run log and it is not a cloud cost ledger.

`runs/` is disposable. Measurements that should survive must be summarized here
or in a promoted model note.

## Current Decision Status

The current Floodgate-like latency baseline is the 2026-05-13 RTX 4000 Ada
YaneuraOu one-game check below.

Short deterministic grids and smoke runs below are historical measurements, not
current deployment evidence.

The strongest current warning is that a 2026-05-10 YaneuraOu workload with
MCTS4096 and evaluation batch size 64 averaged more than 10 seconds per move
request. Most of that wall time was outside model execution.

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

## Current Baseline

Minimum context for future baselines:

- entrypoint: `evaluate_shogi_players.py` or `python -m shogi_arena_agent`
- board backend: `cshogi`
- workload: YaneuraOu, max plies high enough for tail latency
- MCTS simulations per move
- NN leaf eval batch limit
- move time limit
- GPU, vCPU, and PyTorch/CUDA stack

### Floodgate-Like One-Game Check

Purpose: measure one-game, one-move latency behavior. This is different from
self-play throughput, where multiple games can be sharded across worker
processes.

Context:

- Entry point: `evaluate_shogi_players.py`
- Model: d256-h1024-heads8-l6-shogi
- Workload: checkpoint vs YaneuraOu MaterialLv1 `go nodes 1`
- Request: one move decision
- Output unit: MCTS simulation
- Board backend: `cshogi`
- Device: cuda
- GPU: RTX 4000 Ada
- vCPU: 6
- Max plies: 320
- Template: `runpod-torch-v280`
- Torch/CUDA: torch 2.8.0+cu128
- Measured: 2026-05-13

| MCTS simulations per move | NN leaf eval batch limit | Move time limit | Games | Game-level worker processes | Concurrent games per process | Avg request wall | P95 request wall | Max request wall | Avg model calls | Avg model wall | Avg non-model wall | Avg output/sec | Sample count | CPU util avg | CPU util max | GPU util avg | GPU util max | GPU memory max | Result | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1024 | 64 | 9.0s | 1 | N/A | N/A | 2.363s | 4.085s | 5.043s | 23.929 | 1.956s | 0.407s | 481.6 | not measured | not measured | not measured | not measured | not measured | not measured | 0-1, game_over, 84 plies | Stayed below 10s with margin in this one-game check. |
| 2048 | 64 | 9.0s | 1 | N/A | N/A | 4.600s | 8.376s | 9.005s | 56.763 | 3.834s | 0.766s | 502.1 | not measured | not measured | not measured | not measured | not measured | not measured | 0-1, game_over, 76 plies | Stayed below 10s in this one-game check, but tail latency is close to the move budget. |

Notes:

- Both rows used `shogi-arena-agent` main at `ceb702d` and the d256 checkpoint.
- RunPod reported `costPerHr=0.20`, 31 GB RAM, and 6 vCPU for both pods.
- CPU/GPU utilization columns are intended to be populated from
  `gpu_summary.json` produced by the RunPod evaluation wrapper. The first two
  rows predate that sampler.
- `Game-level worker processes` is N/A because this workload is one live game;
  sharding other games would not speed up the current move.
- `Concurrent games per process` is N/A because there is only one current game.
- The relevant batching mechanism is one-tree MCTS leaf batching through
  `NN leaf eval batch limit`.
- `MCTS2048` averaged 2041.0 completed simulations per request because the
  9-second move time limit can stop search before the configured simulation
  count.

## Historical Measurements

These entries preserve measured facts. Do not use short deterministic grids,
short max-plies smoke runs, or older runtime stacks as current Floodgate
deployment evidence.

### Search-Driven Repeated Calls

This path covers inference where a search or planning loop repeatedly chooses
model inputs. MCTS-style play belongs here.

#### Shogi Checkpoint MCTS

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod RTX 4090, CUDA, torch 2.11.0+cu130
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: MCTS32, batch=1 model calls, checkpoint device cuda
- Workload: 4 games vs YaneuraOu `go nodes 1`, max 80 plies

Measured performance:

- `request_wall_time_sec`: avg 0.306s, p95 0.431s, max 0.601s
- `model_call_count`: avg 33.0 per request
- `model_wall_time_sec`: avg 0.184s per request
- `non_model_wall_time_sec`: avg 0.122s per request
- `output_count`: avg 32.0 simulations per request
- `output_per_sec`: avg 111.8 simulations/sec

Notes:

- Historical status: low-simulation baseline; not a current Floodgate candidate
  setting.
- Result: 0-4-0, all game_over, avg 58.0 plies
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-09

#### Shogi Checkpoint MCTS Batched Leaf Evaluation

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod RTX 4090, CUDA, torch 2.9.1+cu128
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: MCTS32, checkpoint device cuda
- Workload: 2 games vs deterministic legal player, max 40 plies

Measured performance:

| Evaluation batch size | Avg model calls | Avg request wall | P95 request wall | Avg model wall | Avg non-model wall | Avg output/sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.000 | 0.219s | 0.295s | 0.120s | 0.099s | 170.8 |
| 4 | 9.225 | 0.157s | 0.239s | 0.058s | 0.098s | 240.6 |
| 8 | 5.389 | 0.152s | 0.247s | 0.052s | 0.100s | 275.6 |
| 16 | 3.325 | 0.156s | 0.280s | 0.046s | 0.110s | 269.5 |

Notes:

- Historical status: batch-mechanism check against a deterministic legal player;
  not a deployment workload.
- Result: no illegal moves; batch 8 had one game end before max plies, so its
  request count was 36 instead of 40.
- Batched evaluation reduced model calls and model wall time substantially, but
  non-model search overhead became the dominant remaining cost.
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-10

#### Shogi Checkpoint MCTS N / Batch Grid

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod RTX 4090, CUDA, torch 2.9.1+cu128
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: checkpoint device cuda
- Workload: 2 games vs deterministic legal player, max 8 plies

Measured performance:

| MCTS simulations | Evaluation batch size | Avg request wall | P95 request wall | Max request wall | Avg model calls | Avg model wall | Avg non-model wall | Avg output/sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 0.092s | 0.297s | 0.297s | 5.00 | 0.057s | 0.035s | 477.9 |
| 32 | 16 | 0.094s | 0.328s | 0.328s | 3.00 | 0.058s | 0.037s | 495.7 |
| 32 | 32 | 0.083s | 0.318s | 0.318s | 2.25 | 0.053s | 0.030s | 581.9 |
| 128 | 8 | 0.281s | 0.465s | 0.465s | 17.00 | 0.132s | 0.150s | 486.3 |
| 128 | 16 | 0.262s | 0.446s | 0.446s | 9.00 | 0.116s | 0.146s | 531.9 |
| 128 | 32 | 0.250s | 0.426s | 0.426s | 5.62 | 0.107s | 0.143s | 558.9 |
| 512 | 8 | 0.998s | 1.140s | 1.140s | 65.00 | 0.426s | 0.572s | 518.3 |
| 512 | 16 | 0.892s | 1.190s | 1.190s | 33.00 | 0.347s | 0.546s | 592.6 |
| 512 | 32 | 0.844s | 1.314s | 1.314s | 17.75 | 0.313s | 0.531s | 633.7 |
| 1024 | 8 | 1.841s | 2.341s | 2.341s | 129.12 | 0.766s | 1.075s | 563.2 |
| 1024 | 16 | 1.798s | 2.368s | 2.368s | 65.00 | 0.664s | 1.134s | 584.8 |
| 1024 | 32 | 1.686s | 2.309s | 2.309s | 34.38 | 0.610s | 1.076s | 627.0 |

Notes:

- Historical status: short deterministic grid; use only for local speed
  direction.
- All measured settings stayed below a 10-second move wall-clock budget in this
  short workload.
- Batch size 32 was fastest for every measured MCTS simulation count in this
  small grid.
- The workload is short and deterministic, so use it for speed direction, not
  playing-strength conclusions.
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-10

#### Shogi Checkpoint MCTS Large-Batch Grid

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod RTX 4090, CUDA, torch 2.4.1+cu124
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: checkpoint device cuda
- Workload: 2 games vs deterministic legal player, max 8 plies

Measured performance:

| MCTS simulations | Evaluation batch size | Avg request wall | P95 request wall | Max request wall | Avg model calls | Avg model wall | Avg non-model wall | Avg output/sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 2.137s | 3.063s | 3.063s | 34.38 | 0.802s | 1.335s | 495.8 |
| 1024 | 64 | 2.015s | 2.744s | 2.744s | 30.88 | 0.755s | 1.260s | 518.3 |
| 1024 | 128 | 2.051s | 2.914s | 2.914s | 30.50 | 0.765s | 1.286s | 511.8 |
| 2048 | 32 | 4.233s | 5.781s | 5.781s | 68.12 | 1.521s | 2.713s | 502.5 |
| 2048 | 64 | 4.095s | 5.252s | 5.252s | 61.00 | 1.473s | 2.622s | 511.2 |
| 2048 | 128 | 4.101s | 5.407s | 5.407s | 59.62 | 1.500s | 2.601s | 511.8 |
| 4096 | 32 | 7.654s | 9.847s | 9.847s | 140.00 | 2.807s | 4.847s | 541.1 |
| 4096 | 64 | 7.607s | 9.357s | 9.357s | 135.50 | 2.808s | 4.798s | 542.7 |
| 4096 | 128 | 7.507s | 9.048s | 9.048s | 135.38 | 2.769s | 4.738s | 549.3 |
| 8192 | 32 | 15.232s | 20.236s | 20.236s | 286.25 | 5.568s | 9.664s | 545.3 |
| 8192 | 64 | 15.136s | 19.705s | 19.705s | 273.88 | 5.474s | 9.662s | 547.8 |
| 8192 | 128 | 15.225s | 20.022s | 20.022s | 271.88 | 5.517s | 9.708s | 545.0 |

Notes:

- Historical status: short deterministic grid; do not treat its MCTS4096 timing
  as current Floodgate evidence.
- This run used a different RunPod image than the preceding grid because some
  hosts could not start the CUDA 12.8 image with their installed NVIDIA driver.
- Batch 64 was fastest at MCTS1024, MCTS2048, and MCTS8192; batch 128 was
  fastest at MCTS4096. The difference among 32/64/128 was small compared with
  the difference from increasing MCTS simulations.
- MCTS4096 with batch 128 averaged 7.507s per move and stayed within the
  10-second move budget in this short workload. MCTS8192 exceeded the budget.
- This short deterministic workload made MCTS4096 look viable for a 10-second
  budget, but the later YaneuraOu workload below exceeded that budget.
- The workload is short and deterministic, so use it for speed direction, not
  playing-strength conclusions.
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-10

#### Shogi Checkpoint MCTS4096 Versus YaneuraOu

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod RTX 4090, CUDA, torch 2.4.1+cu124
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: MCTS4096, evaluation batch size 64, checkpoint device cuda
- Workload: 4 games vs YaneuraOu MaterialLv1 `go nodes 1`, max 80 plies

Measured performance:

- `request_wall_time_sec`: avg 10.887s, p95 22.631s, max 33.782s
- `model_call_count`: avg 115.779 per request
- `model_wall_time_sec`: avg 3.100s per request
- `non_model_wall_time_sec`: avg 7.788s per request
- `output_count`: avg 4096.0 simulations per request
- `output_per_sec`: avg 478.0 simulations/sec

Result:

- 4 games: 1 win, 2 losses, 1 draw by max plies
- Player as black: 0 wins, 1 loss, 1 draw
- Player as white: 1 win, 1 loss
- End reasons: 3 game_over, 1 max_plies
- Average plies: 70.25
- Illegal moves: 0

Notes:

- Historical status: useful CPU-overhead warning, but not a current baseline
  because it used max 80 plies and an older runtime stack.
- This arena-like workload exceeded a 10-second move wall-clock budget on
  average and had a much higher tail than the short deterministic grids.
- The main cost was non-model search overhead, not model wall time.
- The RunPod image was Ubuntu 22.04, so the local YaneuraOu binary could not be
  reused due to GLIBC/GLIBCXX version mismatch. YaneuraOu MaterialLv1 was built
  inside the container before evaluation.
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-10

#### Shogi Checkpoint MCTS2048 Versus YaneuraOu Secure Smoke

Context:

- Inference path: search-driven repeated calls
- Model: d256-h1024-heads8-l6-shogi
- Environment: RunPod secure RTX 5090, CUDA, torch 2.8.0+cu128
- Input shape: shogi position tokens plus legal candidate moves
- Output shape: candidate move logits plus value
- Request: one move decision
- Output unit: MCTS simulation
- Settings: MCTS2048, evaluation batch size 64, checkpoint device cuda
- Workload: 1 game vs YaneuraOu MaterialLv1 `go nodes 1`, max 16 plies

Measured performance:

- `request_wall_time_sec`: avg 4.810s, p95 5.907s, max 5.907s
- `model_call_count`: avg 57.875 per request
- `model_wall_time_sec`: avg 1.684s per request
- `non_model_wall_time_sec`: avg 3.125s per request
- `output_count`: avg 2048.0 simulations per request
- `output_per_sec`: avg 430.2 simulations/sec

Result:

- 1 game: draw by max plies
- Average plies: 16.0
- Illegal moves: 0

Notes:

- Historical status: smoke only; too few plies for tail-latency conclusions.
- This was a shared RunPod runner smoke, not a strength or stable performance
  benchmark.
- The run completed setup, YaneuraOu build, evaluation, output sync, timings
  output, and pod deletion through `runpod-job-runner`.
- The 8 measured move requests stayed within a 10-second move wall-clock
  budget, but the workload is too small for tail-latency conclusions.
- GPU: NVIDIA GeForce RTX 5090
- Measured: 2026-05-10
