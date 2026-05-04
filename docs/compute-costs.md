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
| 2026-05-03 | policy-only full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 2000 | 512 | 11m20s total, 8m25s training | about $0.13 | 3.96 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-value full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 5000 | 512 | 27m20s total, 25m46s training | about $0.31 | value_loss_weight=0.2; 3.23 steps/s; 12.4 GB CUDA max memory. |
| 2026-05-04 | policy-value progress sync fix smoke EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 600 | 512 | 4m35s total, 2m45s training | about $0.05 | value_loss_weight=0.2; 3.63 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-04 | candidate-aware policy-only smoke EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 1000 | 512 | 5m45s total, 4m07s training | about $0.07 | value_loss_weight=0.0; 4.05 steps/s; 8.5 GB CUDA max memory. |
| 2026-05-04 | candidate-aware policy-only comparison EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 3000 | 512 | 15m16s total, 12m52s training | about $0.18 | value_loss_weight=0.0; 3.89 steps/s; 8.5 GB CUDA max memory. |
| 2026-05-04 | candidate-aware value-only smoke EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 1000 | 512 | 16m27s total, 12m12s training | about $0.19 | policy_loss_weight=0.0; value_loss_weight=1.0; 1.37 steps/s; 8.6 GB CUDA max memory. |
| 2026-05-04 | candidate-aware value-only policy-skip smoke EU | estimated | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 1000 | 512 | about 8-12m total | about $0.09-$0.14 | policy_loss_weight=0.0; value_loss_weight=1.0; estimate after skipping policy scoring. |
