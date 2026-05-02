# Compute Costs

This document records estimated and measured compute cost for expensive jobs.
Cost includes runtime, hardware choice, cloud price, setup overhead, and
operational constraints that affect whether a run is worth starting. It is not
a model-quality report or full experiment log.

## Shogi Move-Choice Cache

| Item | Value |
| --- | --- |
| input | Qhapaq KIF games converted to JSONL |
| games | 18,948 |
| examples | 2,460,722 |
| output | `runs/shogi/qhapaq-all-move-choice-examples.jsonl` |
| compressed output | `runs/shogi/qhapaq-all-move-choice-examples.jsonl.zst` |
| compressed size | about 148 MB |
| compute | Modal CPU worker |
| measured runtime | about 8 minutes |
| measured cost | about $0.19 |

## RunPod Shogi Training

The current RunPod recipe uses an RTX 4090 pod as disposable compute, without a
network volume. The compressed cache is copied to the pod, decompressed on
container disk, trained, and the output directory is synced back.

| Run | Steps | Batch size | Runtime | Cost | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| measured smoke | 50 | 512 | about 216 seconds | not separately recorded | Includes setup, transfer, decompression, training, sync, and pod teardown. |
| estimated main run | 5000 | 512 | about 4.5-6.5 hours | about $3-$5 | Uses the same cache and recipe as the smoke run. |

Current command:

```sh
scripts/runpod_train_shogi_move_choice.sh
```

The estimate should be replaced by measured runtime and cost after the 5000-step
run completes.
