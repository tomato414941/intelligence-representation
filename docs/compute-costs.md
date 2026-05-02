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
| train output | `runs/shogi/qhapaq-train-move-choice-examples.jsonl` |
| eval output | `runs/shogi/qhapaq-eval-move-choice-examples.jsonl` |
| train examples | 2,220,818 |
| eval examples | 239,904 |
| train compressed output | `runs/shogi/qhapaq-train-move-choice-examples.jsonl.zst` |
| eval compressed output | `runs/shogi/qhapaq-eval-move-choice-examples.jsonl.zst` |
| compressed size | about 140 MB train, about 16 MB eval |
| compute | Modal CPU worker |
| measured runtime | about 13 minutes for train, about 2 minutes for eval |
| measured cost | about $0.19 for the original full-cache run; split-cache cost not separately recorded |

## RunPod Shogi Training

The current RunPod recipe uses an RTX 4090 pod as disposable compute, without a
network volume. The compressed cache is copied to the pod, decompressed on
container disk, trained, and the output directory is synced back.

| Run | Hardware | Steps | Batch size | Runtime | Cost | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| measured smoke | RunPod RTX 4090 | 50 | 512 | about 216 seconds | not separately recorded | Includes setup, transfer, decompression, training, sync, and pod teardown. |
| aborted main run | RunPod RTX 4090, EU-RO-1, $0.69/hr | 5000 planned | 512 | about 12 minutes | about $0.14 | Stopped manually because the small model did not appear to justify a multi-hour GPU run. |
| estimated main run | RunPod RTX 4090, EU-RO-1, $0.69/hr | 5000 | 512 | about 4.5-6.5 hours | about $3-$5 | Uses train/eval split caches and the current recipe. |

Current recipe:

| Item | Value |
| --- | --- |
| RunPod image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| allowed CUDA versions | `12.8`, `12.9`, `13.0` |
| storage | 80 GB container disk, no network volume |
| max runtime guard | 420 minutes |
| train cache input | `runs/shogi/qhapaq-train-move-choice-examples.jsonl.zst` |
| eval cache input | `runs/shogi/qhapaq-eval-move-choice-examples.jsonl.zst` |
| output directory | `runs/shogi/runpod-qhapaq-split-b512-steps5000` |
| model size knobs | embedding dim 32, hidden dim 64, 4 heads, 1 layer |
| objective knobs | value loss weight 0.2 |
| DataLoader knobs | 4 workers, pinned memory |

Current command:

```sh
scripts/runpod_train_shogi_move_choice.sh
```

The estimate should be replaced by measured runtime and cost after the 5000-step
run completes.
