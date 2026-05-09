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
