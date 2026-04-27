# RunPod Training Notes

This project can train the mixed GPT prototype on CPU by default or on a CUDA GPU
when running on a RunPod container with PyTorch CUDA support.

## Device Selection

The CLI default remains CPU-compatible:

```sh
uv run python -m intrep.train_gpt --max-steps 20
```

On RunPod, use automatic CUDA detection:

```sh
uv run python -m intrep.train_gpt --device auto --max-steps 200
```

Use `--device cuda` only when CUDA must be required. It exits with a CLI error if
`torch.cuda.is_available()` is false.

```sh
uv run python -m intrep.train_gpt --device cuda --max-steps 200
```

## Checkpoint Saving

Write a final checkpoint with `--checkpoint-path`:

```sh
uv run python -m intrep.train_gpt \
  --device auto \
  --corpus file \
  --corpus-path /workspace/data/corpus.jsonl \
  --max-steps 200 \
  --checkpoint-path /workspace/checkpoints/mixed-gpt.pt
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

## Generated Environment Sweep

Run the seed x generated-eval-slice sweep and write one normalized run summary
per cell:

```sh
uv run python -m intrep.experiment_pipeline \
  --output-dir /workspace/runs/generated-environment \
  --seed 7 \
  --seed 13 \
  --max-steps 200
```

The sweep writes:

```text
generated-environment-seed-<seed>-<slice>.json
comparison.json
failures.json
```

`comparison.json` is the metric table for successful runs. `failures.json`
records any failed seed/slice cells so a partial sweep can still be inspected.
Use `--eval-slice` one or more times to run a subset of slices.
