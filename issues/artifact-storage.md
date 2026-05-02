# Artifact Storage

## Issue

Large generated artifacts are currently local-only unless explicitly uploaded
elsewhere. This is risky once a dataset cache, model checkpoint, tokenizer, or
evaluation result becomes an input for future work.

## Artifacts To Consider

| Artifact | Example | Why It Matters |
| --- | --- | --- |
| generated dataset | `qhapaq-all-move-choice-examples.jsonl.zst` | Expensive enough to regenerate that local-only storage is fragile. |
| source-derived records | `qhapaq_all_games.jsonl` | Smaller than full examples and useful for rebuilding task-specific caches. |
| failure logs | `qhapaq-all-move-choice-examples.failures.jsonl` | Explains skipped or invalid source records. |
| model checkpoint | `checkpoint.pt` | Needed to reproduce evaluation and continue training. |
| training metadata | `metrics.json`, config, git commit, command | Needed to know what a checkpoint means. |
| tokenizer or encoding config | text tokenizer files, input encoding versions | Needed to make checkpoints usable. |
| evaluation outputs | arena match JSON, task metrics | Needed to compare checkpoints. |

## Open Question

Choose a long-lived artifact home. Candidate options include Hugging Face Hub,
object storage, or another artifact store. GitHub should remain for code,
documentation, and small metadata rather than large generated files.
