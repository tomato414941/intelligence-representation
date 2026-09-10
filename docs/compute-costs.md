# Compute Costs

This document keeps forward-looking compute-cost references for choosing future
run settings. It is not a complete run history. Keep rows only while they help
estimate or plan likely future runs.

It keeps runtime, hardware, price, throughput, memory, and run size; operational
incidents and model-quality interpretation belong elsewhere.

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

Next time this cache is regenerated on Modal, record the worker CPU count,
memory, image/Python environment, and parallelism settings. Those details were
not recorded for the run above.

## RunPod Shogi Training

RunPod shogi jobs use disposable GPU pods. The current KISS flow avoids network
volumes. This section records cost and runtime only; operational defaults belong
in the training script, and model-quality interpretation belongs in evidence
docs.

RunPod prices are recorded as observed at run time. Check the RunPod console or
pricing page before using these rows for future cost estimates.

| Date | Run | Status | Compute | Model | Data | Steps | Batch | Runtime | Cost | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-03 | policy-only full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 2000 | 512 | 11m20s total, 8m25s training | about $0.13 | 3.96 steps/s; 8.1 GB CUDA max memory. |
| 2026-05-03 | policy-value full-cache baseline workers0 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 5000 | 512 | 27m20s total, 25m46s training | about $0.31 | value_loss_weight=0.2; 3.23 steps/s; 12.4 GB CUDA max memory. |
| 2026-05-04 | candidate-aware policy-only comparison EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 3000 | 512 | 15m16s total, 12m52s training | about $0.18 | value_loss_weight=0.0; 3.89 steps/s; 8.5 GB CUDA max memory. |
| 2026-05-04 | candidate-aware value-only smoke EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 1000 | 512 | 16m27s total, 12m12s training | about $0.19 | policy_loss_weight=0.0; value_loss_weight=1.0; 1.37 steps/s; 8.6 GB CUDA max memory. |
| 2026-05-04 | candidate-aware value-only policy-skip comparison EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | Qhapaq split cache | 3000 | 512 | 24m18s total, 22m03s training | about $0.28 | policy_loss_weight=0.0; value_loss_weight=1.0; 2.27 steps/s; 5.3 GB CUDA max memory. |
| 2026-05-09 | engine-analysis small train EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | shogi engine-analysis bundle, 50 train games / 10 eval games | 500 | 512 | 3m42s total, 2m05s training | about $0.04 | value_loss_weight=0.2; 4.00 steps/s; 7.8 GB CUDA max memory. CPU measured 300 steps in about 14m24s at 0.35 steps/s with batch 128. |
| 2026-05-09 | engine-analysis 1000-game train EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 | shogi engine-analysis bundle, 1000 train games / 20 eval games | 1750 | 512 | 8m30s total, 6m40s training | about $0.10 | value_loss_weight=0.2; early-stopped at step 1750; best eval step 750; 4.52 steps/s; 8.0 GB CUDA max memory. |

## RunPod Shogi Evaluation

| Date | Run | Status | Compute | Players | Games | Search | Runtime | Cost | Notes |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| 2026-05-09 | d32 vs engine-analysis best MCTS8 EU | measured | RunPod RTX 4090, $0.69/hr | d32-h64-heads4-layers1 checkpoint vs d256-h1024-heads8-layers6 checkpoint | 20 | MCTS8 each, CUDA checkpoint inference | 5m50s total, 3m31s evaluation | about $0.07 | GPU confirmed as NVIDIA GeForce RTX 4090. |
| 2026-05-09 | engine-analysis 1000-game vs smoke best MCTS8 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 checkpoint vs d256-h1024-heads8-layers6 checkpoint | 20 | MCTS8 each, CUDA checkpoint inference | 2m47s total, 1m16s evaluation | about $0.03 | GPU confirmed as NVIDIA GeForce RTX 4090. |
| 2026-05-09 | d256 shogi vs YaneuraOu nodes1 MCTS8 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 checkpoint vs YaneuraOu `go nodes 1` | 10 | checkpoint MCTS8 CUDA vs YaneuraOu nodes1 | 2m25s total, 14s evaluation | about $0.03 | GPU confirmed as NVIDIA GeForce RTX 4090. |
| 2026-05-09 | d256 shogi vs YaneuraOu nodes1 MCTS16/32 EU | measured | RunPod RTX 4090, $0.69/hr | d256-h1024-heads8-layers6 checkpoint vs YaneuraOu `go nodes 1` | 20 | checkpoint MCTS16 and MCTS32 CUDA vs YaneuraOu nodes1 | 5m04s total, 2m27s evaluation | about $0.06 | 10 games each for MCTS16 and MCTS32; GPU confirmed as NVIDIA GeForce RTX 4090. |
| 2026-05-10 | shared runner secure smoke vs YaneuraOu nodes1 | smoke | RunPod secure RTX 5090, $0.99/hr | d256-h1024-heads8-layers6 checkpoint vs YaneuraOu `go nodes 1` | 1 | checkpoint MCTS2048 batch64 CUDA vs YaneuraOu nodes1 | 2m56s total, 1m44s remote eval/build command | about $0.05 | Shared runner path completed setup, evaluation, output sync, timings output, and pod deletion; GPU confirmed as NVIDIA GeForce RTX 5090. |

## Shogi Self-Play Generation Throughput

Self-play generation throughput and cost notes are tracked in
`docs/shogi/self-play-generation-throughput.md`.

## Cellular Rule Inference Sizing Reference

For future 6x6 rule-inference runs with up to eight demonstrations and
d256/h1024/heads8/l6, batch 16: the 2026-09-09 A5000 measurement took 247 seconds
for 6000 training steps (about 24 steps/s). The disposable job, including setup,
validation/test evaluation and output retrieval, took 325 seconds. A local
8-vCPU machine with four Torch threads needed 14 seconds for a 20-step probe.

At the [published A5000 Secure Cloud rate of $0.27/hr](https://www.runpod.io/articles/guides/ai-server-cost),
325 seconds corresponds to about $0.025 of GPU time, excluding disk charges.
This is a sizing estimate, not an invoice. Check current pricing before future
runs; context length and board size affect attention cost.

The 2026-09-10 noise/change follow-up used A40 after A5000 had no available
instances. With the same model and batch size, 6000-step training took
202–269 seconds per run. Two clean replications took 611 seconds including
transport and evaluation; two augmented training runs plus five stress
evaluations took 625 seconds. Reevaluating the five fixed checkpoints with an
additional recent-four control took 220 seconds. This makes a small additional
control much cheaper than retraining.

Across those three disposable jobs, elapsed time was 1456 seconds. At the
observed A40 pod rate of $0.49/hr (also listed in the
[published rate guide](https://www.runpod.io/articles/guides/ai-server-cost)),
this is approximately $0.20 of GPU time, excluding disk charges, not an invoice.
Durable timing and resource records accompany the
[experiment artifacts](research/cellular-rule-stress.md#artifacts-and-verification).
