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
| 2026-05-03 | policy-only full-cache smoke | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq shogi move-choice train/eval cache | 50 | 512 | 2m25s total, 7.2s training | about $0.03 | Passed. Full-cache sync/decompress/load, CUDA forward/backward, DataLoader settings, checkpoint, metrics, and output sync worked. Training throughput was 6.94 steps/s and CUDA max memory was about 8.1 GB. |
| 2026-05-03 | policy-only full-cache baseline | failed | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq shogi move-choice train/eval cache | target 2000, reached 350 | 512 | 3m50s total before failure | about $0.04 | Setup, sync, decompression, and training startup worked. Training reached 350/2000 steps at about 7.8 steps/s and 8.1 GB CUDA max memory, then SSH timed out with the pod not responding. No metrics or checkpoint were synced. Follow-up local probing suggests CPU RAM pressure from full Python-object cache plus `num_workers=4` is more likely than CUDA memory exhaustion. |
| 2026-05-03 | policy-only full-cache baseline workers0 | failed | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq shogi move-choice train/eval cache | 2000 | 512 | 58s total before setup failure | about $0.01 | Did not reach training. The pod landed on the same US host/IP as the previous failed run, and `scripts/setup_runpod.sh` failed because CUDA driver initialization reported no usable GPU. No metrics or checkpoint were produced. |
| 2026-05-03 | policy-only full-cache baseline workers0 EU | planned | RunPod RTX 4090, assume $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq shogi move-choice train/eval cache | 2000 | 512 | about 10-20 minutes | about $0.12-$0.25, guarded at about $0.35 | Retries workers0 while pinning the pod to `EU-RO-1`, avoiding the US host that failed the previous two attempts. |

Current recipe:

| Item | Value |
| --- | --- |
| RunPod image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| allowed CUDA versions | `12.8`, `12.9`, `13.0` |
| storage | 80 GB container disk, no network volume |
| max runtime guard | 30 minutes for the 2000-step baseline; 420 minutes default |
| cost guard | no separate cost guard; use the runtime guard and the estimate above |
| train cache input | `runs/shogi/qhapaq-train-move-choice-examples.jsonl.zst` |
| eval cache input | `runs/shogi/qhapaq-eval-move-choice-examples.jsonl.zst` |
| output directory | `runs/shogi/runpod-qhapaq-split-d256-h1024-l6-policy-only-steps2000-workers0-eu-ro` |
| model size knobs | embedding dim 256, hidden dim 1024, 8 heads, 6 layers |
| objective knobs | learning rate 0.0005, value loss weight 0.0 |
| eval knobs | 4096 train-eval examples, 4096 eval examples for the 2000-step baseline |
| DataLoader knobs | 0 workers, pinned memory |

Current command:

```sh
scripts/runpod_train_shogi_move_choice.sh
```

Planned full-cache baseline command:

```sh
DATA_CENTER_IDS=EU-RO-1 \
MAX_STEPS=2000 \
MAX_RUNTIME_MINUTES=30 \
MAX_TRAIN_EVAL_EXAMPLES=4096 \
MAX_EVAL_EXAMPLES=4096 \
NUM_WORKERS=0 \
OUTPUT_DIR=runs/shogi/runpod-qhapaq-split-d256-h1024-l6-policy-only-steps2000-workers0-eu-ro \
scripts/runpod_train_shogi_move_choice.sh
```

The workers0 attempt did not test the DataLoader memory hypothesis because the
pod failed CUDA setup. The next attempt pins `EU-RO-1` to avoid the US host used
by the failed attempts.
