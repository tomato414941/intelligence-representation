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
