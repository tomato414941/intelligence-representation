# From Cellular Prediction To Action

This bounded experiment connects the existing frozen rule-inference Transformer
to action selection. It measures whether experience from executed actions helps
on later tasks with new boards and goals. It does not add a new world framework,
policy trainer, or memory architecture.

## Question And Scope

After several executed interventions in an unfamiliar cellular world, does the
model choose better interventions on new tasks than the same model with its
experience removed?

The model has previously learned prediction across rules. At evaluation time its
weights stay fixed. The private evaluation rule is absent from training and all
earlier cellular evaluations supplied as exclusions. Experience is a short list
of observed transitions, carried across tasks under that same rule.

This is one-step model-based control. The planner, available interventions, goal
utility, and experience buffer are explicit. The learned component is the
context-conditioned prediction of cellular dynamics. There is no claim of
learned planning, information-seeking exploration, long-term memory, or transfer
to an unseen rule family.

## Task

Each independent trial contains nine tasks in the same private world:

1. Receive a new random 6x6 board and an independently sampled binary goal.
2. Consider 37 interventions: wait, or flip one of the 36 cells.
3. Predict the board after each intervention followed by one unknown-rule step.
4. Choose the action with the highest expected number of cells matching the goal.
5. Execute that action and observe its result.
6. Carry only the executed before/after transition into the next task.

Each task starts from a fresh board; this is a sequence of experiments on a
stable world, not navigation along one board trajectory. Goals need not be
exactly reachable. Evaluation measures the quality of the chosen action relative
to the best available action, rather than claiming arbitrary goals are solvable.

The goal is not generated from a privileged future outcome. It is not passed
into the predictor: the explicit planner applies the known utility to predicted
cell probabilities. For this additive utility, per-cell marginals suffice to
calculate expected utility; no independence assumption about cells is needed.

Intervention mechanics are known: a flip directly changes an input cell. The
subsequent cellular update rule must be inferred. No local neighborhood counts,
rule table, rule ID, or unexecuted transition enters the neural predictor.

## Experience And Leakage Controls

- Distinct tasks within a trial have initial boards separated by Hamming
  distance at least three. Their entire sets of one-flip candidate boards are
  therefore disjoint. A later query cannot repeat any earlier executed input.
- Candidate pools, goals and tie-breaking permutations are generated before
  actions and independently of the private rule. The same task arrays and
  ordering are used across model seeds, verified by a task hash.
- The experience policy chooses actions. Its own executed inputs and observed
  outputs are the only new demonstration pairs added to its history.
- All learned action choices are fixed before the evaluator enumerates true
  counterfactual outcomes to calculate regret.
- The checkpoint was trained with context lengths 0/1/4/8. At intermediate
  rounds use the largest of those lengths available, taking the most recent
  examples. The primary curve reads rounds 0, 1, 4 and 8, where every previous
  transition is included. This avoids changing model training for this test.

## Decision Controls

All decision controls see the same current board, goal and action candidates.
Controls using experience receive the rollout policy's experience. They are not
separate agents collecting their own histories.

| Method | Evidence and decision rule |
| --- | --- |
| experience | Frozen Transformer with observed executed transitions; maximize predicted expected goal utility. |
| forgetful | Same checkpoint and planner, empty context on every task. |
| wrong context | Same demonstrated inputs, outputs replaced using a distinct donor rule; a diagnostic of reliance on relevant experience. |
| family Bayes | Same real evidence, plus explicit knowledge of the local cellular rule family; unknown rule bits have probability 1/2 and B0 is fixed to zero. |
| no-op | Always let the board evolve without flipping a cell. |
| random | Exact expected performance of uniformly choosing among the 37 actions. |
| oracle | Best true available utility, calculated only by the evaluator. |

The family baseline is privileged by its handcrafted rule-family structure; it
is not evidence that the neural model receives those features. Wrong-context
outputs and oracle outcomes are evaluation controls, not observations available
to the experience policy.

## Measurement And Fixed Protocol

Primary endpoint: after eight executed tasks, the paired reduction in regret
relative to the forgetful model on the ninth, previously unseen task. Regret is
the difference in matching-cell count between the best true action and the
selected action. Secondary endpoints are optimal-action rate (all ties count as
correct), paired effects against wrong context, and curves at 0/1/4/8 experiences.

Only tasks whose true action utilities are unequal enter the headline metrics.
All-tie tasks are counted separately; they cannot establish useful control.
Static cells shared by every outcome contribute nothing to regret or action
optimality. Random performance is calculated exactly, avoiding sampling noise.

- Checkpoints: the existing clean model seeds 31, 32 and 33, with unchanged
  d256/h1024/heads8/l6 weights. No new training or checkpoint selection.
- Engineering pilot: two rule pairs, one nine-task trial per world;
  rule/data seeds 49101/53001. It checks execution and runtime, not quality gates.
- Final evaluation: 64 independent true/donor rule pairs, four nine-task trials
  per true rule; rule/data seeds 59101/63001. Exclude training identities,
  original clean validation/test rules, stress validation/test rules and pilot
  rules. Check all exclusions by identity, not only random seed.
- Report model seeds separately. Bootstrap 2000 independent rule pairs, keeping
  their trials together, for 95% intervals and paired differences.
- Retain the first trial of the first four rule pairs as illustrative traces,
  chosen before outcomes are inspected. They are examples, not the evidence
  behind the aggregate claim.
- Use float32 inference and bounded candidate batches. Changing inference batch
  size for memory use does not change the experimental task or training budget.

The protocol is fixed before the final action results are inspected. Successful
control here would support using learned context-dependent predictions for
decisions. It would not establish the project's broader representation system.

## Run

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python \
  -m intrep.problems.cellular_rule_inference.control \
  --checkpoint models/cellular-rule-followup-20260910/clean-seed31/checkpoint.pt \
  --device cuda \
  --exclude-rules-from models/cellular-rule-followup-20260910/clean-seed31/validation.json \
  --exclude-rules-from models/cellular-rule-followup-20260910/clean-seed31/test.json \
  --exclude-rules-from models/cellular-rule-followup-20260910/stress-validation.json \
  --exclude-rules-from models/cellular-rule-followup-20260910/clean-seed31/stress-test.json \
  --exclude-rules-from models/cellular-control-20260910/pilot.json \
  --output runs/cellular-control-20260910/clean-seed31.json
```

For RunPod use the existing disposable command runner and setup procedure in
[RunPod](../runpod.md). Retain needed evaluation JSON and traces outside `runs/`.
Checkpoints remain in their existing durable model directories.

## Results (2026-09-10)

Protocol and implementation were committed as `9dfe0ed` before the final
evaluation. All three existing clean checkpoints completed the same 64-world,
four-trial protocol. Each model executed 2304 tasks; 256 tasks occur at each
round. At the primary endpoint, one task had identical utilities for all
actions, leaving 255 decision tasks. Training weights were unchanged.

### Optimal Action Selection

| Previously executed tasks | Seed 31 | Seed 32 | Seed 33 |
| ---: | ---: | ---: | ---: |
| 0 | 14.84% | 8.20% | 16.02% |
| 1 | 56.64% | 57.81% | 60.94% |
| 4 | 89.06% | 89.84% | 90.62% |
| 8 | 91.76% | 94.12% | 93.33% |

Different rows contain different tasks. The paired evidence is the comparison
with controls on the same final tasks:

| On the ninth task | Seed 31 | Seed 32 | Seed 33 |
| --- | ---: | ---: | ---: |
| Experience: optimal action | 91.76% | 94.12% | 93.33% |
| Forgetful: optimal action | 13.73% | 7.06% | 13.33% |
| Wrong context: optimal action | 10.59% | 9.80% | 9.41% |
| Known-family baseline: optimal action | 98.82% | 98.82% | 98.82% |
| Random action: expected optimal rate | 5.99% | 5.99% | 5.99% |
| Experience: mean regret, cells | 0.090 | 0.067 | 0.094 |
| Forgetful: mean regret, cells | 2.804 | 3.231 | 2.882 |

The primary paired regret reduction against forgetting was:

- Seed 31: **2.714 cells**, 95% rule-pair bootstrap interval **[2.398, 3.059]**.
- Seed 32: **3.165 cells**, interval **[2.824, 3.510]**.
- Seed 33: **2.788 cells**, interval **[2.463, 3.135]**.

These intervals resample evaluation worlds, not training runs. The model seeds
are separate replications. They share task inputs but can accumulate different
executed histories because their actions differ.

### Interpretation

The existing predictions are useful for decisions: a frozen Transformer can
use its own previous executed transitions to select a near-best intervention
on a new board with a new goal. Empty and wrong-world contexts remove most of
that advantage. This extends the earlier prediction-accuracy result to measured
control performance under an explicit one-step planner.

The known-family baseline remains stronger: it has 98.82% optimal action rate
and 0.012-cell mean regret after eight experiences. The learned model is not
shown to be a better inference algorithm than this privileged reference.
Experience is useful within the trained cellular rule family; the result does
not establish transfer between different kinds of problems or long-horizon
planning. The choice to collect an observation is not learned or optimized for
information gain.

## Artifacts And Verification

Durable outputs are in `models/cellular-control-20260910/`:

- `clean-seed31.json`, `clean-seed32.json`, `clean-seed33.json`: metrics,
  bootstrap intervals, rule identities and recorded action traces.
- `pilot.json`: the engineering pilot, excluded from the final worlds.
- `manifest.json`: checkpoint references, hashes and evaluation provenance.
- `control.png`, `.pdf`, `.svg`, `.csv`: scientific figure and plotted values.
- `replay.html`: self-contained interactive replay, with no model execution or
  network dependency in the browser.
- `runpod_timings.json` and resource records: the completed A40 evaluation job.

The existing checkpoints remain in
`models/cellular-rule-followup-20260910/clean-seed{31,32,33}/checkpoint.pt`.
All checkpoint hashes matched the evaluation records. All three evaluations
have identical task hashes and identical held-out rule identities. Every final
rule identity is disjoint from training and all declared earlier evaluations.

Seven new tests check intervention semantics, disjoint candidate pools,
probability-based action choice, inference-batch alignment, executed-only
experience, resistance to static scoring gains, frozen reproducibility and
identity exclusions. The full suite passed **479 tests**. Python lint passed.
The replay was checked in a browser at desktop and mobile widths, including
experience selection, model/world switching and browser errors.

The disposable A40 job completed in 189 seconds, with 142 seconds in the
evaluation workload. Its pod was deleted after outputs were retrieved.

Rebuild the displays from the retained measurements:

```sh
uv run --no-project --with matplotlib python scripts/plot_cellular_rule_control.py \
  --artifact-dir models/cellular-control-20260910
uv run python scripts/render_cellular_control_replay.py \
  models/cellular-control-20260910/clean-seed31.json \
  models/cellular-control-20260910/clean-seed32.json \
  models/cellular-control-20260910/clean-seed33.json \
  --figure models/cellular-control-20260910/control.png \
  --output models/cellular-control-20260910/replay.html
```
