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

RunPod shogi jobs use disposable RTX 4090 pods without network volumes. This
section records cost and runtime only; operational defaults belong in the
training script, and model-quality interpretation belongs in evidence docs.

| Date | Run | Status | Compute | Model | Data | Steps | Batch | Runtime | Cost | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-03 | policy-only full-cache smoke | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 50 | 512 | 2m25s total, 7.2s training | about $0.03 | Passed; 6.94 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-only full-cache baseline | failed | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | target 2000, reached 350 | 512 | 3m50s total before failure | about $0.04 | Pod stopped responding over SSH; no metrics/checkpoint synced. |
| 2026-05-03 | policy-only full-cache baseline workers0 | failed | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 2000 | 512 | 58s total before setup failure | about $0.01 | Same US host as prior failure; CUDA setup failed. |
| 2026-05-03 | policy-only full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 2000 | 512 | 11m20s total, 8m25s training | about $0.13 | Passed on EU-RO-1; 3.96 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-value full-cache baseline workers0 EU | estimated | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 5000 | 512 | about 25-30m total | about $0.30-$0.35 | Planned with value_loss_weight=0.2 on EU-RO-1. |
