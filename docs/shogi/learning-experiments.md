# Shogi Learning Experiments

This document records facts from shogi learning experiments. It is not a
complete run log and it does not treat `runs/` as a durable source of truth.

## Scope

Record:

- training run conditions
- training data amount and source mix
- strength evidence for the trained checkpoint
- known limits that affect strength interpretation

Do not record transient compute acquisition attempts or full shard-level JSON
payloads.

## Training Runs

| Experiment | Date | Method | Starting checkpoint | Next-cycle checkpoint | Cycles | Experience sources per cycle | Environment | Remote job time |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| `shogi-learning-20260515-001` | 2026-05-15 | online replay | `models/d256-h1024-heads8-l6-shogi/checkpoint.pt` | cycle final checkpoint | 4 | `self:64`, `usi:64` | RTX 4000 Ada, 16 vCPU, 62 GiB, secure, EUR-IS-1, $0.26/hr | 42.9 min |

`Next-cycle checkpoint` records which checkpoint from one cycle is used as the
starting checkpoint for the next cycle. `cycle final checkpoint` means
`cycle-N/checkpoint.pt`, not `cycle-N/best-checkpoint.pt`.

## Generation Settings

| Experiment | Generation worker processes | Concurrent games per process | MCTS simulations per move | NN leaf eval batch limit | Max plies | USI opponent | USI side assignment |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `shogi-learning-20260515-001` | 8 | self-play: 8; USI: 1 effective | 16 | 32 | 320 | YaneuraOu | checkpoint black, USI white |

## Optimization Settings

| Experiment | Optimizer steps/cycle | Batch size | Learning rate | Policy loss weight | Value loss weight | Num workers | Early stopping | Eval during training |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `shogi-learning-20260515-001` | 1,000 | 512 | 0.0001 | 1.0 | 1.0 | 0 | not wired | disabled |

## Replay And Evaluation Settings

| Experiment | Replay capacity | Min replay size | Replay sample size/cycle | Steps per replay sample pass | Effective passes/cycle | Eval ratio for generated games | Training eval source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `shogi-learning-20260515-001` | 131,072 | 8,192 | 8,192 | 16 | 62.5 | 0.05 | fixed eval selection |

`Steps per replay sample pass` is `replay_sample_size / batch_size`.
`Effective passes/cycle` is `optimizer_steps_per_cycle / steps_per_replay_sample_pass`.

## Training Data

| Experiment | Replay seed selection | Replay seed examples | Fixed eval selection | Fixed eval examples | Generated games | Generated train examples | Generated eval examples | Training eval used | Final replay size |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `shogi-learning-20260515-001` | `data/shogi/training-data-bundles/online-replay-seed-20260512/data-selection.json` | 17,644 | same data-selection artifact | 5,967 | 512 | 65,992 | 3,473 | fixed eval, not generated eval | 83,636 |

## Cycle Metrics

| Experiment | Cycle | Generated train examples | Generated eval examples | Replay size after append | Sampled examples | Optimizer steps | Train loss before | Train loss after | Fixed eval loss before | Fixed eval loss after | Best eval loss | Best eval step | Fixed eval accuracy after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shogi-learning-20260515-001` | 1 | 16,493 | 588 | 34,137 | 8,192 | 1,000 | 2.4425 | 0.0293 | 2.5003 | 7.7066 | 2.5003 | 0 | 0.3566 |
| `shogi-learning-20260515-001` | 2 | 16,209 | 891 | 50,346 | 8,192 | 1,000 | 4.0264 | 0.0287 | 7.7066 | 9.6335 | 7.7066 | 0 | 0.3255 |
| `shogi-learning-20260515-001` | 3 | 17,544 | 906 | 67,890 | 8,192 | 1,000 | 3.8232 | 0.0302 | 9.6335 | 8.9047 | 8.9047 | 1000 | 0.3377 |
| `shogi-learning-20260515-001` | 4 | 15,746 | 1,088 | 83,636 | 8,192 | 1,000 | 3.7654 | 0.0315 | 8.9047 | 10.9028 | 8.9047 | 0 | 0.3057 |

## Generated Experience Summary

| Experiment | Source | Games | Average plies | Wins by checkpoint side | Wins by opponent side | Draws / max-plies | Generated train examples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shogi-learning-20260515-001` | self-play | 256 | 220.7 | 63 black wins | 84 white wins | 109 | included in 65,992 total |
| `shogi-learning-20260515-001` | checkpoint-vs-USI | 256 | 50.6 | 0 checkpoint black wins | 256 USI white wins | 0 | included in 65,992 total |

## Inference Batch Observations

| Experiment | Source | Configured NN leaf eval batch limit | Actual NN leaf eval batch size avg | Fill ratio avg | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `shogi-learning-20260515-001` | self-play | 32 | about 6.1-6.4 | about 0.19-0.20 | Batches did not fill close to the configured limit. |
| `shogi-learning-20260515-001` | checkpoint-vs-USI | 32 | about 14.3-14.5 | about 0.45 | USI games ended quickly and used effective concurrent games per process of 1. |

## Strength Evidence

| Experiment | Evidence | Before | After | Result | Conclusion |
| --- | --- | ---: | ---: | --- | --- |
| `shogi-learning-20260515-001` | Fixed eval loss | 2.5003 | 10.9028 | worsened | Not evidence of strength improvement. |
| `shogi-learning-20260515-001` | Initial-vs-final checkpoint match | not run | not run | unavailable | Strength improvement is undetermined. |

## Observations

- This run is evidence that the online replay training path can produce
  checkpoints, but it is not evidence that the final checkpoint is stronger.
- The fixed eval loss worsened from the initial checkpoint to the final
  checkpoint.
- No direct initial-vs-final match was run.
- The generated data had known structure issues: fixed-side USI games and many
  self-play `max_plies` draws.
- Early stopping existed in the training implementation but was not wired
  through the online replay config for this run.
- Training progress was not emitted during each cycle.

## Follow-Up Issues

- `issues/shogi-usi-generated-experience-side-balance.md`
- `issues/shogi-online-replay-training-progress-visibility.md`
- `issues/shogi-online-replay-training-config-boundary.md`
- `issues/shogi-online-replay-strength-evaluation.md`
- `issues/shogi-online-replay-step-budget-policy.md`
- `issues/shogi-self-play-max-plies-draw-quality.md`
- `issues/shogi-generated-eval-responsibility.md`
- `issues/shogi-nn-leaf-eval-batch-fill.md`
