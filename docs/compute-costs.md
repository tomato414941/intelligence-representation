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

The subsequent fixed-checkpoint control evaluation used A40 for three models,
each with 64 worlds, four nine-task trials per world and 37 action candidates.
Including empty-context and wrong-context decision controls, the workload
evaluated 710,400 model queries in 142 seconds. Total disposable-job time was
189 seconds, or about $0.026 of GPU time at the published $0.49/hr rate above,
excluding disk charges. No training was required. Timing and resource records
are retained with the [control results](research/cellular-rule-control.md).

## Recurrent Multimodal Agent Sizing Reference

The 2026-09-10 integrated agent used one A40 at the observed $0.49/hr rate.
The 5.10-million-parameter model (d256/h1024/heads8/l6, 32 memory vectors)
jointly learned action values, text and image/audio/feedback forecasts, unrolling
six observations per sampled episode. With batch 16, bf16 and four Torch CPU
threads, 3,000 updates took 1,469 seconds, about 2.04 updates/s. A subsequent
100-update mixed-replay run, including some eight-action episodes, took 52 seconds.
Memory peaked at 2,548 MiB on the GPU; sampled GPU utilization averaged 23%.

The complete job, including setup, evaluation, actor experience collection,
retrieval and deletion, took 1,656 seconds. At the rate above this is about
$0.225 of GPU time, excluding disk charges. The earlier 16-update execution
probe took 76 seconds end to end, bringing the two jobs to about $0.236 of GPU
time. These are sizing estimates rather than invoices. The recurrent unroll
and several shared-core calls per step make these updates much more expensive
than the single-pass cellular predictor. Detailed timings are retained with
the [multimodal agent artifacts](multimodal-agent.md#artifacts-and-verification).

## Rejected Two-Model Language Integration

For the 2026-09-11 Qwen3-4B NF4 + learned native predictor configuration,
rank-eight LoRA and batch size two, the measured RTX A6000 job used an observed
RunPod rate of $0.53/hour. One hundred joint updates took 1,058 seconds
(10.58 seconds/update). Twenty subsequent updates mixing actual actor experience
took about three minutes. These batches reconstruct complete six-step episodes
and also supervise two conversations.

Including setup, before/after validation, eight actual interaction episodes,
output retrieval and pod deletion, the job took 1,692 seconds (28.2 minutes),
about $0.25 at the observed hourly rate. Peak allocated CUDA memory during the
initial training/evaluation process was 42.42 GB; the job resource monitor saw
42,819 MiB GPU memory in use. This records the measured workload on a 48 GB GPU. Model and validation details are
in [the historical two-model experiment](research/language-agent-qwen-20260911.md).

## Single-Core Language Correction

The replacement 5.10-million-parameter shared native model used one RTX A5000
at the observed $0.27/hour rate on 2026-09-11. Six hundred joint updates took
218.6 seconds (0.36 seconds/update) with batch size two. Each update replayed
complete native episodes and supervised two byte-level conversations. Fifty
additional updates mixed actual actor experience with retained experience.

The full disposable job, including setup, evaluation, actor collection, output
retrieval and pod deletion, took 332.2 seconds, about $0.025 of GPU time excluding
disk charges. Peak sampled GPU memory was 1,813 MiB. These figures describe the
[corrected single-core experiment](language-agent.md), whose general language
quality remains insufficient; they are not sizing estimates for acquiring
reliable language ability.

## Joint Full-Parameter LFM Sizing Reference

The 2026-09-11 nine-source LFM comparison used one A40 at the observed pod rate
of $0.49/hour. Both models started from their pinned official bases and trained
all attached parameters with FP32 AdamW, learning rate `1e-5`, and four CPU
threads. Each update accumulated all four text sources (64 targets each), three
image sources (one image each), one shogi position and one native transition
with its preceding history. These small batches do not exhaust the available
training populations.

| Assembly | Updates | Time in optimizer updates | Seconds/update | Peak Torch CUDA allocation |
| --- | ---: | ---: | ---: | ---: |
| LFM2.5-230M, 233.35M attached parameters | 1,000 | 508.0 s | 0.508 | 4,346 MiB |
| LFM2.5-350M, 358.14M attached parameters | 1,000 | 533.2 s | 0.533 | 6,254 MiB |

The complete sequential job took 1,825.1 seconds (30m25s), including provisioning,
transfer of the complete recipe inputs, setup, five fixed-panel evaluations per
model, generation/input controls, checkpoint retrieval and pod deletion. At the
observed rate this is about $0.248, excluding disk charges; it is an estimate,
not an invoice. The external GPU monitor peaked at 7,324 MiB and averaged 44.2%
utilization during the monitored remote workload. These measurements describe
this recipe and reference convolution implementation; the earlier CPU SGD
execution checks are not a controlled hardware speed comparison.

The [learning report](joint-learning-evaluation.md) records improvements and
language deterioration. Timing, environment and resource records are retained
with the durable `models/joint-lfm-learning-20260911/` artifacts. Future training
budgets should use both capability/retention measurements and these workload
costs, rather than extrapolating quality from falling aggregate loss.

## Twelve-Source Question Learning Sizing Reference

The 2026-09-11 fixed/varied-question comparison used one A40 at the observed
$0.49/hour rate. Both conditions used a single LFM2.5-350M body, 358.23M attached
trainable parameters, FP32 AdamW at `1e-5` and four Torch CPU threads. Each update
read 128 positions from each of four text streams, two records from each image,
speech, sensor and shogi source, two native transitions with history, and one
complete BoolQ passage/question. Varied alternated original and added objectives.

| Condition | Updates | Measured training time | Seconds/update | Peak Torch CUDA allocation |
| --- | ---: | ---: | ---: | ---: |
| Fixed objectives | 3,889 | 3,600.6 s | 0.926 | 7,942 MiB |
| Varied objectives | 4,047 | 3,600.1 s | 0.890 | 7,943 MiB |

Training measurements include input reading and question construction, forward/
backward passes and optimizer steps; they exclude checkpoint and evaluation I/O.
The complete sequential job took 9,137.5 seconds (2h32m17s), about $1.244 of
GPU time at the observed rate, excluding disk charges. The two full final
checkpoints total about 8.6 GB. The final retrieval took 247.6 seconds; the fixed
checkpoint had already been copied during varied training. Peak monitored GPU
memory was 8,972 MiB and mean utilization was 65.0% over the remote workload.

These are workload sizing measurements, not a claim that the training budget
produces a generally capable model. The [comparison report](question-learning-evaluation.md)
records narrow improvements and substantial instruction-response deterioration.
Timing and resource samples accompany the durable
`models/question-learning-20260911/` artifacts. Full generated-answer evaluation
and checkpoint retrieval are material overheads when planning another run.

## Instruction Retention And Remote Checkpoint Storage

The 2026-09-11 three-condition retention pilot used one A40 at the observed
$0.49/hour rate, one LFM2.5-350M body, 358.23M trainable attached parameters,
FP32 AdamW at `1e-5` and four Torch CPU threads. Each condition ran 300 joint
updates with all twelve sources. Assistant supervision retained up to 2,048
conversation tokens with 1,024-token overlap; other source batch sizes matched
the twelve-source comparison above.

| Conversation condition | Updates | Measured training time | Seconds/update | Peak Torch CUDA allocation |
| --- | ---: | ---: | ---: | ---: |
| 128-token all-role stream | 300 | 275.9 s | 0.920 | 7,956 MiB |
| Assistant targets with context | 300 | 286.9 s | 0.956 | 11,713 MiB |
| Assistant targets, source weight 8 | 300 | 288.2 s | 0.961 | 11,713 MiB |

The optimizer-update time totals 851.0 seconds; it excludes full generated-answer
evaluation, checkpoint serialization, CPU restoration and storage transfer.
Each full AdamW checkpoint is about 4.3 GB. The job uploads it to project R2
storage and downloads the object for byte comparison before starting the next
condition. This sequence keeps the GPU allocated during storage work, so
training time alone substantially understates its cost.

The complete disposable job took 3,539.5 seconds (58m59s), including
provisioning, input transfer, evaluation, CPU verification, R2 upload/readback,
result retrieval and pod deletion. At the observed rate this is about $0.482
of GPU time, excluding disk charges, not an invoice. Peak externally sampled
GPU memory was 15,167 MiB; mean GPU utilization was 27.0% over the monitored
remote workload, including storage waits. Future budgets should account for
this storage overhead as well as the 14m11s spent in optimizer updates.

Three checkpoints occupy about 12.9 GB. At the
[published R2 Standard rate of $0.015/GB-month](https://developers.cloudflare.com/r2/pricing/),
keeping them for a full month is about $0.20 before account-level free allowances
and request charges. This is a storage estimate, not an invoice; check current
rates and existing account usage when planning retention. The
[comparison report](instruction-retention-evaluation.md) and its durable
`reports/instruction-retention-20260911/` artifacts contain model-quality,
environment, archive and timing records.

## Shared Prediction Execution Efficiency

Local checks on 2026-09-13 compared the pre-change implementation at
`da724fedaefdbcae8d110a70533cd5c6e3dd8efc` with batched original objectives and
faster source-state hashing. Measurements used CPU, four Torch threads and
PyTorch 2.11.0. The forward/backward fixture was an untrained 16-dimensional,
three-layer LFM body: these are component timings, not LFM2.5-350M or A40
throughput estimates.

| Component | Before | After | Local speedup |
| --- | ---: | ---: | ---: |
| Source-state digest, 71,445 distinct-record entries | 116.3 ms | 14.8 ms | 7.8x |
| MNIST original objective, eight images | 55.5 ms | 12.7 ms | 4.4x |
| Text original objective, eight 128-token blocks | 57.2 ms | 13.2 ms | 4.4x |
| Sensor classification, eight 128-sample windows | 59.8 ms | 14.5 ms | 4.1x |
| Speech classification, eight short unequal-length fixtures | 54.9 ms | 16.9 ms | 3.3x |

The digest test used synthetic sampler values sized from the completed run and
required identical hashes; values are medians of nine repetitions. Batching
used the same model and inputs for each comparison, with seven repetitions and
alternating order. MNIST used actual 28x28 training images with four-pixel
patches; the other inputs were synthetic fixtures. Forward/backward timings
exclude input-file I/O and optimizer updates. At two records per batch, these
four component speedups were 1.6–1.7x. Tests also compare losses
and gradients, including unequal-length speech weighted equally per example.

Joint training now collects detached loss scalars once per device; finite-loss
and finite-gradient checks still run before optimizer updates. Source metrics
can still cause other CUDA synchronizations. The image-follow-up runner can
overlap CPU archive verification and transfer with remaining GPU work, with
`--isolate-timing` available for processing-time comparisons. The complete-model
GPU comparison below measures their combined effect. The component ratios cannot
be multiplied or used directly as GPU billing reductions.

### Complete-Model GPU Comparison (2026-09-14)

One A40 ran all 358,228,139 trainable parameters of the existing LFM2.5-350M
rule-transfer model, in float32 with AdamW. The comparison used PyTorch
2.8.0+cu128, Transformers 5.17.0, eager attention and four Torch CPU threads.
It restored the same 4.3 GB checkpoint, optimizer, sampling cursors and RNG for
each trial. All twelve data sources and four supplemental lessons remained
enabled, with their complete training populations and original loss weights.

The old code at `da724fedaefdbcae8d110a70533cd5c6e3dd8efc` and optimized code at
`a3fb0c81080fe3aa01591d72854e7fc5b8782f4e` ran in old/new/new/old order, each with
eight warmup updates and 32 measured updates. Timings include input reads, all
loss computations, forward/backward, AdamW, CUDA synchronization and source-state
hashing. They exclude model restoration, correctness snapshots, evaluation and
checkpoint I/O. The common harness is
`scripts/benchmark_shared_prediction_training.py` at
`8cbf7afd56c15329305f5a33b81443d378fa4cf6`.

| Implementation and batch | Seconds / 32 updates | Original-batch data equivalents / second | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Old, original batch, mean of two runs | 57.62 | 0.555 | 7.81 GiB |
| Optimized, original batch, mean of two runs | 51.43 | 0.622 | 7.81 GiB |
| Optimized, all batches x2, one run | 87.71 | 0.730 | 10.03 GiB |
| Optimized, all batches x4, one run | 164.27 | 0.779 | 14.45 GiB |

At the unchanged batch, the optimized code reduced complete-update time by
10.7% (1.120x throughput). The two old measurements were 57.13 and 58.11 seconds;
the two optimized measurements were 50.94 and 51.92 seconds. Initialization,
per-update source and lesson traces, and sequence-position counts matched.
The maximum loss difference divided by `max(1, abs(old_loss))` was 5.27e-5,
within the predeclared 1e-4 tolerance.
The first update used added question forms and had exactly equal parameters and
clipped gradients. A separate check exercised all twelve original objectives on
the second update: relative L2 differences were 1.82e-10 for all parameters and
2.30e-7 for active clipped gradients, below the declared 1e-6 and 1e-4 limits.

For the batch sweep, every source's records per update and every supplemental
lesson batch increased together. Data equivalents count this common multiplier;
they do not count optimizer updates. The x4 candidate processed 25.2% more data
per second than the optimized original batch, and 40.3% more than the old code.
Moving from x2 to x4 added only 6.8%, so no larger batch was measured. Larger
batches consumed different sample spans and had one trial each; these results
do not establish learning quality or cost to reach a target score. The x4 peak
PyTorch memory reservation was 16.39 GiB, including its cache; the table reports
allocated tensors. Required memory also depends on sampled sequence lengths
and CUDA context overhead.

The archive comparison used one sequential/overlap pair. Each schedule ran a
fresh training process with eight warmup and 64 measured updates, plus CPU
restoration, R2 upload and full download verification of the same existing
4.3 GB checkpoint. Both retained identical training inputs and loss traces.

| Schedule | Measured training, 64 updates | Complete training and archive schedule |
| --- | ---: | ---: |
| Sequential | 103.88 s | 817.30 s (13m37s) |
| Overlapped | 104.77 s | 694.60 s (11m35s) |

Overlap saved 122.71 seconds (15.0%) in this pair, while measured training slowed
by 0.86%. Archive-process durations were 660.74 and 691.82 seconds, so storage
time itself varied. Complete schedule times include process/model restoration,
warmup, correctness checks and temporary-copy cleanup. They exclude initial
worker provisioning and source restoration; checkpoint serialization was not
remeasured. One pair does not establish a general saving across networks or
workloads, and this ratio must not be multiplied by the training speedup.

The next throughput candidate uses four records per text/conversation/BoolQ
source, eight per other source, and supplemental lesson batches of
`[64, 32, 32, 32]`. Keep the source weights, float32, AdamW and learning rate
`1e-5` unchanged, and overlap one completed-checkpoint archive with remaining
GPU work. Changing batches requires an explicit new experiment initialized
from the checkpoint because exact resume validates the original recipe. Compare
learning at matched exposure before adopting this as the default for further
learning experiments. No capability evaluation was run during this profiling.

The disposable A40 allocation lasted 2,929.26 seconds (48m49s), including initial
setup recovery, all model restores, measurements, archive checks, supplementary
gradient checks, result retrieval and deletion. At the observed $0.49/hr rate,
the GPU estimate is $0.399, excluding disk and storage request charges, not an
invoice. The worker and both temporary R2 checkpoint copies were deleted; the
original trained checkpoint remains in its existing archive. The local report
directory is `reports/rule-transfer/efficiency-gpu-20260914/`, including
`selected-training-settings.json`. Small evidence files are retained at project
R2 prefix `shared-prediction/efficiency-20260914/results-1605`; no profiling
checkpoint weights are retained.

### Further Cost Reduction Before Another Learning Run (2026-09-15)

Prioritize fewer CPU/GPU waits and remaining question batching at unchanged
training settings, then measure the unused CUDA implementations. Keep all
358,228,139 parameters trainable, all twelve full source populations and the
four lessons. This review reuses completed experiments and inspects code;
it does not add a GPU run or establish another speedup.

The [completed rule-transfer experiment](rule-transfer.md#computation-and-retention)
provides the relevant allocation baseline:

| Work | Hours | GPU cost estimate at the recorded $0.49/hour |
| --- | ---: | ---: |
| Training updates, all ten endpoints | 6.98 | $3.42 |
| Setup, evaluation, saving, verification and other waits | 3.68 | $1.80 |
| Complete worker allocation | 10.66 | $5.22 |

Non-training work accounted for 34.5% of allocation, including necessary
evaluation and preservation. Existing timers do not split that entire amount
into storage, evaluation and idle time. As sensitivity calculations only,
halving this portion would save 17.3% of the whole job, about $0.90; reducing
training time by another 10% would save 6.5%, about $0.34. Neither is a forecast.
The previously measured 10.7% training reduction and archive overlap are already
implemented; their gains must not be counted again as new opportunities.

#### Execution Changes To Investigate First

1. **Defer question metrics and batch compatible questions.**
   `QuestionSource.loss` still converts GPU metrics with `float(value)` for
   each question before backward. Collect detached metrics on the device and
   transfer them together when needed, retaining finite-loss/gradient checks
   before every update. Added forms still call `_score` separately; original
   assistant conversation, BoolQ, shogi and native objectives also remain
   sequential. Start with compatible lengths and output heads, preserving
   each question's loss weight, target mask, sampler order and question count.
   Tokenization can be reused for immutable prompts, but learned embeddings
   cannot be reused across parameter updates.

   Existing traces provide a diagnostic, not a prediction of removable time:

   | Optimized batch | Original-form seconds/update | Added-form seconds/update | Original / added body calls per update |
   | --- | ---: | ---: | ---: |
   | x1, mean of two trials | 1.566 | 1.648 | 18.88 / 19.13 |
   | x2, one trial | 2.516 | 2.966 | 25.88 / 34.25 |
   | x4, one trial | 4.625 | 5.642 | 39.06 / 64.50 |

   Each trial contributes 16 measured updates of each form category. Larger
   batches consume different samples, and call counts do not identify GPU
   compute time. They show why increasing batch counts alone leaves substantial
   sequential execution. The x4 throughput gain over x2 was only 6.8%.

2. **Measure CUDA implementations while retaining FP32.**
   Every GPU trial logged the reference `causal_conv1d_fn` fallback. The model
   has ten convolution blocks and six attention blocks; no convolution share
   of elapsed time has been measured. The
   [CUDA causal-convolution implementation](https://github.com/Dao-AILab/causal-conv1d)
   supports FP32 and the model's width-three convolution, making this a concrete
   candidate. Verify the pinned Transformers 5.17.0 integration, backward
   results and CPU paths before deployment; importing the optional package
   changes the implementation selected by its wrapper.

   `JointTrainer` also forces `foreach=False` for AdamW and gradient clipping.
   [PyTorch 2.8 AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html)
   documents faster foreach/fused candidates and additional foreach memory.
   Test these separately, preserving all optimizer moments, clipping and loss
   weights. Restored optimizer parameter groups include execution options,
   so changing the constructor alone is insufficient. GPU savings and numerical
   equivalence remain unmeasured for this model.

3. **Shorten evaluation and the final storage wait.**
   Record separate times for serialization, CPU restore, upload, full readback
   and evaluation. The existing image runner overlaps archives, but the
   intervention runner still archives its endpoints sequentially after
   evaluation. Overlap completed immutable checkpoints with remaining planned
   GPU work where possible. The last archive still lies on the completion path.

   The current `rclone check --download` reads complete content; its `--checkers`
   setting controls concurrent file checks. Increasing it does not split one
   large checkpoint into parallel parts. A candidate is a multithreaded copy
   into temporary storage followed by complete local byte comparison, which
   needs an additional checkpoint-sized disk allocation. Keep CPU restoration
   and full-content verification. The
   [check documentation](https://rclone.org/commands/rclone_check/) and
   [multithread transfer documentation](https://rclone.org/docs/#multi-thread-cutoff)
   describe these distinct paths. Existing 661--692 second archive timings
   include multiple stages, so they cannot establish this candidate's saving.

   Generated-answer evaluation also recomputes the complete prefix for every
   token in `generate_text`, `answer_loss` and `MeasuredReadout` because the
   shared LFM core uses `use_cache=False`. Evaluate batching and fresh
   per-question inference caches while preserving prompts, EOS handling,
   query counts and every scored answer. LFM needs both convolution and
   attention state; this is an inference change, with no cache carried across
   training updates. [Transformers caching](https://huggingface.co/docs/transformers/kv_cache)
   explains the avoided prefix computation. Its benefit for these mostly short
   answers remains unmeasured. Keep the evaluation panels and scheduled gates.

#### Changes Requiring A Learning Comparison

The x4 batch remains a throughput candidate: 25.2% more data per second than
the optimized x1 batch, with no demonstrated reduction in cost to reach a
quality target. Compare per-source and per-question exposure as well as update
counts and retention before adoption. Changing the recipe requires a new
experiment fork, as exact resume checks its original settings.

TF32 matmul is disabled, and training has no BF16 autocast. Either may reduce
compute time, but changes numerical behavior. Test separately from batching;
[TF32](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
changes matmul input precision, while
[autocast](https://docs.pytorch.org/docs/2.8/amp.html)
selects precision per operation. Autocast with FP32 parameters and AdamW states
does not by itself halve the 4.3 GB checkpoint. SDPA is supported by the
[pinned LFM implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/lfm2/modeling_lfm2.py),
but the current eager-attention path and any replacement need a measured
comparison on actual input lengths and dtypes.

The x4 trial's 16.39 GiB peak reservation makes 24 GB GPUs worth a fit and cost
comparison, without establishing that all future sequence lengths fit. Choose
using current quotes and complete workload cost, including preparation and
storage waits. Hourly price or peak arithmetic throughput alone is insufficient.

#### Next Decision

Prepare the metric/batching changes and their loss, gradient and sampling
comparisons locally. For a later GPU comparison, restore the existing checkpoint
and all source/optimizer/RNG state, use the original batch first and repeat
baseline/candidate trials in balanced order. Cover original and added forms;
reuse the previous declared tolerances for behavior-preserving changes. Measure
complete updates and allocation time, and test only candidates prepared in
advance on one disposable worker. A training-only comparison does not require
repeating the large checkpoint-upload benchmark or preserving profiling weights.

Before provisioning, verify the complete source-file and dependency manifests;
the previous two allocations needed recovery for omitted inputs or code.
Set the total comparison budget using setup, restore, verification and cleanup
as well as timed updates. At the historical rate, 30 minutes of A40 allocation
would cost $0.245 before storage; this is arithmetic, not a current quote or an
authorized run budget.

Further quality training should use a fixed development decision and retention
criteria. No image-tuition arm reached the previous target despite near-perfect
support scores, so additional updates alone have no established cost benefit.
Do not repeat calibration or the full multi-arm experiment merely to benchmark
execution. Account-level persistent storage should be reviewed separately from
these disposable-worker costs; network storage can remain billable with no
running GPU, as described in [RunPod storage](https://docs.runpod.io/storage/network-volumes).

Offline derivation and input hashes are in
`reports/rule-transfer/cost-review-20260915/{analyze.py,analysis.json}`. Inputs are
the existing final allocation/training report and six GPU comparison traces.
