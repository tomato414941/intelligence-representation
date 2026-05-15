# Shogi Online Experience Replay

This document records facts from shogi online experience replay runs. It is not
a complete run log and it does not treat `runs/` as a durable source of truth.

## Scope

Record:

- online experience replay run conditions
- generated experience source mix
- replay and training settings
- cycle-level data volume
- cycle-level training outcomes
- observed data-quality or orchestration issues

Do not record transient RunPod acquisition attempts or full shard-level JSON
payloads.

## Measurements

| Case | Date | GPU | Pod vCPU/RAM | Cloud | Data center | Rate | Wall time | Initial checkpoint | Sources | Cycles | Replay seed examples | Fixed eval examples | Replay sample size | Max steps/cycle | Batch size | Early stopping | Final checkpoint policy | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `online-replay-runpod-20260515-174845` | 2026-05-15 | RTX 4000 Ada | 16 vCPU, 62 GiB | secure | EUR-IS-1 | $0.26/hr | 43.97 min | `d256-h1024-heads8-l6-shogi/checkpoint.pt` | `self:64`, `usi:64` | 4 | 17,644 | 5,967 | 8,192 | 1,000 | 512 | not wired | final | Completed. USI source was fixed-side checkpoint black vs USI white. |

### Cycle Summary

| Case | Cycle | Appended examples | Replay size | Sampled examples | Training skipped | Train initial loss | Eval initial loss | Eval final loss | Best eval loss | Best eval step | Self-play avg plies | Self-play max-plies draws | Self-play plies/sec | USI avg plies | USI result | USI plies/sec |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `online-replay-runpod-20260515-174845` | 1 | 16,493 | 34,137 | 8,192 | false | 2.4425 | 2.5003 | 7.7066 | 2.5003 | 0 | 218.95 | 29 / 64 | 105.67 | 47.94 | white 64-0 | 132.66 |
| `online-replay-runpod-20260515-174845` | 2 | 16,209 | 50,346 | 8,192 | false | 4.0264 | 7.7066 | 9.6335 | 7.7066 | 0 | 217.22 | 27 / 64 | 110.00 | 49.97 | white 64-0 | 169.56 |
| `online-replay-runpod-20260515-174845` | 3 | 17,544 | 67,890 | 8,192 | false | 3.8232 | 9.6335 | 8.9047 | 8.9047 | 1000 | 235.78 | 29 / 64 | 112.58 | 52.50 | white 64-0 | 207.26 |
| `online-replay-runpod-20260515-174845` | 4 | 15,746 | 83,636 | 8,192 | false | 3.7654 | 8.9047 | 10.9028 | 8.9047 | 0 | 211.03 | 24 / 64 | 101.44 | 52.00 | white 64-0 | 222.87 |

## Observations

- The full RunPod job passed and produced checkpoints for all four cycles.
- Training was not skipped in any cycle because the replay buffer was seeded
  before online generation.
- The USI source produced terminal games in all cycles, but it was fixed-side:
  checkpoint as black and YaneuraOu as white. YaneuraOu won every USI game.
- Self-play produced many `max_plies` draws under the 320-ply cap.
- Early stopping existed in the training implementation but was not wired
  through the online replay CLI/config/RunPod wrapper for this run.
- Training progress was silent during each cycle; liveness required checking
  process state or GPU utilization separately.

## Follow-Up Issues

- `issues/shogi-usi-generated-experience-side-balance.md`
- `issues/shogi-online-replay-training-progress-visibility.md`
- `issues/shogi-online-replay-training-config-boundary.md`
