# Shogi Model Precision And Quantization

Status: open. Priority: medium.

## Issue

Shogi tensor-cache storage size and shogi model precision are separate
problems. Cache dtype compaction reduces artifact size without changing model
semantics. Model precision and quantization affect training memory, inference
latency, checkpoint size, and possibly value calibration.

The project should decide which model-side precision modes are worth supporting
for shogi policy/value training and MCTS inference.

## Desired Shape

Evaluate model precision as an explicit model/runtime decision.

Initial candidates:

- BF16 or FP16 inference for MCTS evaluation
- BF16 training on supported GPUs
- FP32 checkpoint as training source of truth, with optional inference-export
  checkpoints
- later INT8 or weight-only quantized inference if value calibration survives

Evaluation should distinguish:

- training stability
- policy loss and value loss
- value calibration
- fixed-node MCTS strength
- fixed-time MCTS strength
- checkpoint size and inference throughput

## Non-Goals

- combine this with tensor-cache storage dtype work
- introduce 4-bit or INT8 quantization before a baseline full-precision model
  exists
- treat policy accuracy alone as sufficient evidence

## Notes

The first model-side precision change to consider is BF16/FP16 inference or
training, not aggressive quantization. INT8/4-bit quantization should be judged
after a useful full-precision checkpoint exists.
