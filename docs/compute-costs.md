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

| Run | Hardware | Steps | Batch size | Runtime | Cost | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| measured smoke | RunPod RTX 4090 | 50 | 512 | about 216 seconds | not separately recorded | Includes setup, transfer, decompression, training, sync, and pod teardown. |
| estimated main run | RunPod RTX 4090, EU-RO-1, $0.69/hr | 5000 | 512 | about 4.5-6.5 hours | about $3-$5 | Uses the same cache and recipe as the smoke run. |

Current recipe:

| Item | Value |
| --- | --- |
| RunPod image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| allowed CUDA versions | `12.8`, `12.9`, `13.0` |
| storage | 80 GB container disk, no network volume |
| max runtime guard | 420 minutes |
| cache input | `runs/shogi/qhapaq-all-move-choice-examples.jsonl.zst` |
| output directory | `runs/shogi/runpod-qhapaq-all-b512-steps5000` |
| model size knobs | embedding dim 32, hidden dim 64, 4 heads, 1 layer |
| objective knobs | value loss weight 0.2 |

Current command:

```sh
scripts/runpod_train_shogi_move_choice.sh
```

The estimate should be replaced by measured runtime and cost after the 5000-step
run completes.
