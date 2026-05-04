# Compute Costs

This document records estimated and measured compute cost to choose practical
run settings. It keeps runtime, hardware, price, throughput, memory, and run
size; operational incidents and model-quality interpretation belong elsewhere.

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

RunPod shogi jobs use disposable RTX 4090 pods without network volumes. This
section records cost and runtime only; operational defaults belong in the
training script, and model-quality interpretation belongs in evidence docs.

| Date | Run | Status | Compute | Model | Data | Steps | Batch | Runtime | Cost | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-03 | policy-only full-cache smoke | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 50 | 512 | 2m25s total, 7.2s training | about $0.03 | 6.94 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-only full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 2000 | 512 | 11m20s total, 8m25s training | about $0.13 | 3.96 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-value full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 5000 | 512 | 27m20s total, 25m46s training | about $0.31 | value_loss_weight=0.2; 3.23 steps/s; 12.4 GB CUDA max memory. |
| 2026-05-04 | policy-value progress artifact smoke EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 1600 | 512 | 25m41s total, 21m01s training | about $0.30 | value_loss_weight=0.2; 1.27 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-04 | policy-value progress sync fix smoke EU | estimated | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 600 | 512 | under 20m total | about $0.10-$0.20 | value_loss_weight=0.2; checkpoint_every=500; metrics_every=500; keep_last_n=2. |
