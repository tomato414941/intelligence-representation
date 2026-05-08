# Artifact Storage

Status: closed.

## Issue

Large generated artifacts are currently local-only unless explicitly uploaded
elsewhere. This is risky once a dataset cache, model checkpoint, tokenizer, or
evaluation result becomes an input for future work.

## Artifacts To Consider

| Artifact | Example | Why It Matters |
| --- | --- | --- |
| generated cache | `training-view-cache.pt` | Expensive enough to regenerate that local-only storage is fragile. |
| source-derived records | `qhapaq_all_games.jsonl` | Smaller than full examples and useful for rebuilding task-specific caches. |
| failure logs | `qhapaq_all_games_failures.jsonl` | Explains skipped or invalid source records. |
| model checkpoint | `checkpoint.pt` | Needed to reproduce evaluation and continue training. |
| training metadata | `metrics.json`, config, git commit, command | Needed to know what a checkpoint means. |
| tokenizer or encoding config | text tokenizer files, input encoding versions | Needed to make checkpoints usable. |
| evaluation outputs | arena match JSON, task metrics | Needed to compare checkpoints. |

## Open Question

Choose a long-lived artifact home. Candidate options include Hugging Face Hub,
object storage, or another artifact store. GitHub should remain for code,
documentation, and small metadata rather than large generated files.

## Resolution

This broad issue was closed because it mixed several artifact responsibilities.
Follow-up work is split into narrower issues:

- [`../model-artifact-policy.md`](../model-artifact-policy.md): what belongs
  with long-lived model checkpoints under `models/`.
- [`../source-derived-artifact-policy.md`](../source-derived-artifact-policy.md):
  how to store source-derived records and failure logs.
- [`../evaluation-artifact-policy.md`](../evaluation-artifact-policy.md): where
  evaluation metrics and match outputs belong.
- [`../long-lived-artifact-home.md`](../long-lived-artifact-home.md): whether
  large long-lived artifacts should stay local or use an external artifact home.

Existing narrower issues also cover parts of the original list:

- [`../shogi-training-view-tensor-cache.md`](../shogi-training-view-tensor-cache.md)
  tracks shogi Training View cache format.
- [`../text-tokenizer-policy.md`](../text-tokenizer-policy.md) tracks tokenizer
  workflow and saved tokenizer expectations.
