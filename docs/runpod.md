# RunPod Training Notes

This project can train the Signal text prototype on CPU by default or on a CUDA GPU
when running on a RunPod container with PyTorch CUDA support.

## Device Selection

The CLI default remains CPU-compatible:

```sh
uv run python -m intrep.train_signal_text --train-path /workspace/data/signals.jsonl --max-steps 20
```

On RunPod, use automatic CUDA detection:

```sh
uv run python -m intrep.train_signal_text --train-path /workspace/data/signals.jsonl --device auto --max-steps 200
```

Use `--device cuda` only when CUDA must be required. It exits with a CLI error if
`torch.cuda.is_available()` is false.

```sh
uv run python -m intrep.train_signal_text --train-path /workspace/data/signals.jsonl --device cuda --max-steps 200
```

## Checkpoint Saving

Write a final checkpoint with `--checkpoint-path`:

```sh
uv run python -m intrep.train_signal_text \
  --device auto \
  --train-path /workspace/data/signals.jsonl \
  --max-steps 200 \
  --checkpoint-path /workspace/checkpoints/signal-text-gpt.pt
```

The checkpoint contains:

```text
schema_version
model_state_dict
model_config
training_config
result
```

Checkpoint resume is intentionally not supported yet. The saved state dict is
moved to CPU before writing so the file can be inspected or loaded on non-GPU
machines.

## Future Prediction Evaluation

Use `intrep.evaluate_future_prediction` for generated observation/action
future-prediction checks.
