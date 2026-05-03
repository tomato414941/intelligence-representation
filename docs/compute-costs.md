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

| Date | Run | Status | Compute | Model | Data | Steps | Batch | Runtime | Cost | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-03 | policy-only full-cache smoke | planned | RunPod RTX 4090, assume about $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq shogi move-choice train/eval cache | 50 | 512 | about 10-20 minutes | about $0.12-$0.25, guarded at about $0.35 | Verifies full-cache sync/decompress/load, CUDA forward/backward, DataLoader settings, checkpoint, metrics, and output sync before a longer baseline. |

Current recipe:

| Item | Value |
| --- | --- |
| RunPod image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| allowed CUDA versions | `12.8`, `12.9`, `13.0` |
| storage | 80 GB container disk, no network volume |
| max runtime guard | 30 minutes for the 50-step smoke; 420 minutes default |
| cost guard | no separate cost guard; use the runtime guard and the estimate above |
| train cache input | `runs/shogi/qhapaq-train-move-choice-examples.jsonl.zst` |
| eval cache input | `runs/shogi/qhapaq-eval-move-choice-examples.jsonl.zst` |
| output directory | `runs/shogi/runpod-qhapaq-split-d256-h1024-l6-policy-only-steps50` |
| model size knobs | embedding dim 256, hidden dim 1024, 8 heads, 6 layers |
| objective knobs | learning rate 0.0005, value loss weight 0.0 |
| eval knobs | 1024 train-eval examples, 1024 eval examples for the 50-step smoke |
| DataLoader knobs | 4 workers, pinned memory |

Current command:

```sh
scripts/runpod_train_shogi_move_choice.sh
```

Planned full-cache smoke command:

```sh
MAX_STEPS=50 \
MAX_RUNTIME_MINUTES=30 \
MAX_TRAIN_EVAL_EXAMPLES=1024 \
MAX_EVAL_EXAMPLES=1024 \
OUTPUT_DIR=runs/shogi/runpod-qhapaq-split-d256-h1024-l6-policy-only-steps50 \
scripts/runpod_train_shogi_move_choice.sh
```

The estimate should be replaced by measured runtime and cost after the smoke
run completes.
