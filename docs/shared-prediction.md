# Exchangeable Heads And Joint LFM Learning

2026-09-11. The implementation keeps one learned body while input and output
heads can be attached, detached or replaced. LFM2.5-230M and 350M are independent
size comparisons: each assembled model has exactly one LFM body. The earlier
small `LanguageAgentModel` experiments remain separate historical baselines.

The follow-up [learning evaluation](joint-learning-evaluation.md) completed
1,000 full-model updates per size with all nine sources. Fixed panels show lower
losses across all sources, but generation probes expose language deterioration
and native image forecasts remain worse than persistence. The report includes
per-task metrics and input-omission controls. The three-update figures below
describe the earlier execution check.

The subsequent [twelve-source question comparison](question-learning-evaluation.md)
adds several questions per dataset and about eight times the earlier consumed
data. Text extraction and some reconstruction tasks improve, while general
instruction responses and several relation/history tests remain inadequate.

The [instruction retention pilot](instruction-retention.md) compares that
conversation objective with assistant-only supervision over longer context,
then increases its source loss weight. All conditions retain one trainable
body and the same twelve sources, with frequent development generation probes
and a separate fresh prompt group. The [completed results](instruction-retention-evaluation.md)
show that neither change preserved instruction responses in this short run;
fresh extraction gains coexist with substantial loss of previously correct answers.

## The Learning Contract

```text
source-native records -> exchangeable input heads
                     -> [batch, sequence, hidden]
                     -> one shared learned body
                     -> exchangeable output heads -> predictions / actions
actual targets and outcomes -> per-source losses
                           -> accumulated joint gradient -> full-model update
```

Language, images, audio and boards are examples, not an exhaustive type list.
Numerical sensors, tool results or another observation/action space can supply
their own input head, output head and training source. The core owns neither
those formats nor their losses. A new head must learn how to use the body;
attaching an untrained projection does not confer a new ability immediately.

The original LFM text embedding and output projection remain tied to the same
parameter. They are moved outside the body without changing text computation.
The body retains LFM's causal convolution/attention structure. Observation
tokens precede prediction queries; forecast targets are never input values.
An action query precedes the executed action, which forecasts may condition on.
No second pretrained vision, audio or language model is called.

All attached parameters remain trainable. `JointTrainer` rejects frozen
parameters and requires every registered source in each update. It computes
one mini-batch loss per source, accumulates weighted gradients, clips the joint
gradient and makes one optimizer step. Source computations are sequential to
limit activation memory; mathematically the update uses their joint weighted
loss. It is not necessary to put every record in one batch.

Every declared training population remains in the stream when a source is
added. Training and evaluation are separate. File cursors, pending text tokens,
shuffle orders and episode transitions resume from checkpoints. The current
recipe traverses complete populations without a permanent small-sample limit;
their sizes and exact train/evaluation boundaries are in [datasets.md](datasets.md).
Equal source weights are a starting configuration, not a claim that all tasks
learn at an appropriate rate or that forgetting is eliminated.

## Changing Heads

`SharedPredictor` has no fixed modality enumeration:

```python
import torch
from torch import nn
from intrep.representation.cores.lfm import load_lfm

model = load_lfm("models/lfm2.5-230m")
body = model.core
model.attach_input("sensor", nn.Linear(7, model.dimension))
model.attach_output("forecast", nn.Linear(model.dimension, 4))
hidden = model(model.encode("sensor", torch.randn(1, 8, 7)))
prediction = model.decode("forecast", hidden)

old_head = model.detach_output("forecast")
model.attach_output("forecast", nn.Linear(model.dimension, 2))
assert model.core is body
model.attach_output("forecast", old_head)
```

The next `JointTrainer.step()` synchronizes the attached parameters: new heads
enter its optimizer, detached parameters leave, and the body's optimizer
history survives. Call `synchronize_parameters()` explicitly to do this before
the next step. Existing source callbacks must still match the current heads.

`module_state_dict()` stores the body, each head and parameter ties separately.
`load_module_state_dict()` restores them into explicitly constructed modules
after checking identities, dimensions and ties. The CLI reconstructs its heads
from the data recipe. For a different head architecture, register its constructor
in an extension; the checkpoint does not guess Python classes from tensor shapes.

## Sources And Checkpoints

`intrep.problems.shared_prediction.sources.register_source(kind, configure_heads,
source_factory)` adds a source type without editing the core or trainer.
The factory receives `(model, tokenizer, config, root)` and returns an object
with `loss()`, `state_dict()`, `load_state_dict()` and `provenance()`; optional
`progress()` provides a JSON-compatible summary. The source owns its raw format,
sampling and objective. Extensions also own validation of their custom splits.

Use `--extension package.module` to explicitly import a registration module.
Exact resume requires the same modules; `--extend` permits appending modules
while preserving the existing list. Extensions may not replace a registered
source type. Recipe extension must retain every previous source configuration
and data checksum. This guards against accidentally switching to only the new
task's data.

A checkpoint contains LFM configuration, attention backend, modular weights,
optimizer state, full source states, recipe, data hashes and RNG states. Its
adjacent tokenizer is required when loading. The writer atomically replaces
`checkpoint.pt`; periodic checkpoints default to every 100 joint updates.
Failed updates do not apply a partial source gradient; resume from the last
saved checkpoint after an interrupted run.

## Run

Install the optional dependencies locally:

```sh
uv sync --extra torch --extra vision --extra lfm
```

The downloaded official native checkpoints are:

| Model | Local base | Pinned revision |
| --- | --- | --- |
| LFM2.5-230M | `models/lfm2.5-230m` | `40cb2ad3b3044d5a41eee083a6103c8b523afa45` |
| LFM2.5-350M | `models/lfm2.5-350m` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |

Each base directory retains its original card, license and download provenance.
Prepare the full conversation branches once from the existing raw archive:

```sh
uv run python scripts/prepare_joint_conversations.py \
  --archive data/language/agent-conversations-oasst1-20260911/messages.jsonl.gz \
  --output data/language/joint-conversations-oasst1-20260911
```

The other recipe paths refer to existing project datasets. The command below
is an execution check, with three joint updates and all nine sources per update:

```sh
uv run python scripts/train_shared_prediction.py \
  --base models/lfm2.5-230m --recipe configs/joint-lfm.json \
  --data-root . --output models/joint-lfm-230m-check \
  --steps 3 --device cpu --threads 2 --optimizer sgd \
  --learning-rate 0.0001 --audit-gradients
```

Resume using a **total** update count. Optimizer settings are restored from the
checkpoint; the optimizer kind and gradient-clipping configuration must match:

```sh
uv run python scripts/train_shared_prediction.py \
  --resume models/joint-lfm-230m-check/checkpoint.pt \
  --recipe configs/joint-lfm.json --data-root . \
  --output models/joint-lfm-230m-check --steps 6 --optimizer sgd
```

To add data/heads, provide an expanded recipe and `--extend`; the original
sources remain present. Use a new output directory to keep the prior run intact.
Existing file contents cannot be replaced during exact continuation; new data
belongs in a newly registered source. Base weights and trained checkpoints live
under ignored `models/`, outside disposable `runs/`.

## Evidence And Limits

Both downloaded models have been exercised on CPU with FP32, two threads, SGD
without momentum, learning rate `1e-4` and joint gradient clipping at 1. Each
text source contributes 64 next-token targets per update; each image/board
source contributes one example, and native experience contributes one transition
with its complete preceding history. Image patches are 4×4 and audio chunks
are 128 samples. These are small execution settings, not a throughput-tuned
full training schedule.

| Measurement, three joint updates | 230M assembly | 350M assembly |
| --- | ---: | ---: |
| Trainable parameters, including attached heads | 233,350,242 | 358,141,026 |
| Body layers reached by every source | 14 / 14 | 16 / 16 |
| Body parameter tensors with changed hashes | 130 / 131 | 146 / 147 |
| Seconds per joint update, excluding evaluation/checkpoint I/O | 9.87–9.97 | 14.98–15.56 |
| Peak process RSS, MiB | 3,753 | 4,795 |

These CPU measurements use the reference convolution implementation and are
execution observations, not isolated hardware benchmarks. Training a full
population at these tiny batch sizes would require far more computation than
this check. Quality training needs an appropriate batch/token budget and compute.

The local result directories are `models/joint-lfm-{230m,350m}-20260911/`.
They contain `result.json`, `steps.jsonl`, `provenance.json`, the complete recipe,
`checkpoint.pt`, `tokenizer/` and `reload-verification.json`. Every source produced finite nonzero gradients
in the operator and feed-forward block of every body layer. All parameters
were trainable; finite-precision updates need not alter every tensor bit on
every step. Parameter hashes record which tensors actually changed.

Tests verify original text-logit equivalence after separating heads, tied
weights, causality, gradients from different head pairs, summed-loss update
equivalence, preservation of body optimizer moments during head replacement,
all-source enforcement, source coverage, future-target exclusion and exact
checkpoint continuation. A small LFM run resumed at one update matches a
straight two-update run bit for bit; extension also retains old source cursors.
The two full-size checkpoints were also reloaded: saved text generation matched,
text weights stayed tied, and exchanging both numerical input and output heads
preserved the body's object identity and every parameter hash.

Three updates are not a completed epoch over these populations. The fixed
before/after evaluation currently uses one example/block per source and two
short text prompts. It establishes execution, not generalization, retained
conversation quality, useful image/audio understanding or playing strength.
The 230M Japanese ice-melting response remains factually wrong after this short
run. Long joint training, larger independent evaluation and interference across
tasks remain necessary before making capability claims. This learner currently
replays recorded native experience; it does not itself collect new tool or
environment interactions.
