# Inference Performance

This document catalogs wall-clock inference and interaction performance. It is
not a run log and it is not a cloud cost ledger.

`runs/` is disposable. Measurements that should survive must be summarized here
or in a promoted model note.

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

## Catalog

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

- All measured settings stayed below a 10-second move wall-clock budget.
- Batch size 32 was fastest for every measured MCTS simulation count in this
  small grid.
- The workload is short and deterministic, so use it for speed direction, not
  playing-strength conclusions.
- GPU: NVIDIA GeForce RTX 4090
- Measured: 2026-05-10
