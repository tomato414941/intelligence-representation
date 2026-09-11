# Multimodal Predictive Agent

This page records the initial native-modality experiment. Its text output was
trained on color labels, not general conversation. The
[single-core assembly](language-agent.md) adds conversation training to the same
core; its results must be evaluated separately.

This implementation connects language, images, audio, actions, feedback and
recurrent memory to one learned Transformer. It is an integrated research
prototype of the [project direction](predictive-representation-system.md).

## Data And Computation

```text
text / image / audio / previous action / actual feedback
  -> input embeddings + previous learned memory
  -> shared Transformer
  -> updated memory
  -> action choice / text / predicted image, audio and feedback

executed action -> environment -> actual observation and feedback
  -> append source episode
  -> next inference update

selected source episodes -> replay buffer -> joint learning -> checkpoint
```

The model owns the recurrent update and all prediction heads. An environment
executes actions and returns observations; its hidden state does not enter the
model. Training targets and teacher actions are separate from model inputs.
Source records, explicit training selection, replay sampling and inference
memory have separate responsibilities.

All supported inputs have real encoders. Images use patches with coordinate
features rather than a table for a fixed board. Audio uses waveform chunks and
time coordinates. Language uses UTF-8 bytes. Dynamic input shapes are an API
property; generalization to unseen sizes or unrestricted speech is measured
separately from accepting those inputs.

Inference memory is a bank of learned vectors updated by the shared core. It
can be reset or saved with a session. Training unrolls updates through an
episode. Replay contains episodes so that observations from unrelated worlds
are not silently treated as one memory history. The existing uniform
`ReplayBuffer` supplies training samples.

Action scores learn from teacher choices where those labels exist and from
off-policy TD targets using actual rewards and a slowly updated target network.
Actor-generated episodes have no privileged teacher labels. They still train
action values, image/audio prediction and feedback prediction. A `cycle` command
collects episodes, writes a new explicit selection, mixes them with retained
data, learns a new checkpoint and repeats. Inference memory starts afresh at
each episode boundary; replay reconstructs it from the recorded inputs under
the current model weights.

New checkpoints retain the cumulative training-source history across weight
initialization, even when replay selections are replaced. Evaluation checks
that history rather than treating omitted old replay episodes as unseen data.

This use of replay and a separate target network follows the established
[DQN approach](https://doi.org/10.1038/nature14236); it is not a novelty claim.

## Grounded Integration Task

A navigation world supplies an image containing an agent and two colored
markers. Text specifies which marker to reach. An audible cue specifies the
current control mode. Later images arrive without repeating those instructions,
so the model must carry their information in memory. Instructions and control
modes can change during an episode, accompanied by new text or sound.

The actor chooses an action itself. Its outcome predictor forecasts the next
image, sound and feedback before execution. The environment then supplies the
actual result, which becomes the next input and a durable experience record.
Text output reports the target being pursued. This task gives each input a
specific job instead of attaching redundant modalities to a classification
example.

Audio cues are generated waveforms with a defined meaning in this world. The
audio interface also reads and writes waveform files. Performance on these cues
does not measure unrestricted speech recognition or speech synthesis. Text
training in this task has a small vocabulary and does not establish general
conversation ability.

## Verification

Checks cover variable input sizes, audio and text paths, recurrent gradients,
masking and target isolation, memory reset, source persistence, explicit splits,
replay sampling, checkpoint restoration and actual actor-environment execution.
End-to-end measurements report what the trained checkpoint can do, independently
of whether a particular interface has been implemented.

The input diagnostic replays identical held-out histories with empty text,
blank images, silent audio, or memory reset at every observation. It measures
the frozen model's dependence on those inputs, including the distribution
shift from perturbation. It does not substitute for training matched models
without a modality. Forecast measurements also include copying the previous
image and predicting silence as simple reference predictions.

```sh
uv run python scripts/evaluate_multimodal_inputs.py \
  --checkpoint models/multimodal-agent-20260910/cycle/round-000/learning/checkpoint.pt \
  --selection data/multimodal-navigation-20260910/selection.json \
  --output models/multimodal-agent-20260910/input-diagnostics.json
```

## Initial Integration Run

The initial integration run uses the following fixed settings:

| Item | Setting |
| --- | --- |
| Model | 5,095,054 parameters; d256 / h1024 / 8 heads / 6 layers |
| Recurrent memory | 32 vectors of dimension 256 |
| Inputs | UTF-8 text, RGB pixels, 128-sample waveform chunks, action and feedback embeddings |
| Data | 1,024 training, 64 validation and 128 test episodes; disjoint initial layouts |
| Recorded episode length | 6 actions |
| Pretraining | 3,000 updates; batches of 16 whole episodes; seed 41 |
| Joint losses | Teacher action CE, reward TD, text CE, image/audio MSE, feedback regression and flag BCE |
| Experience cycle | 16 additional actor episodes of 8 actions; epsilon 0.15; 100 mixed-replay updates |
| Model selection | Fixed final update, without choosing a checkpoint using test scores |

All training episodes are replayed from their first observation. The control
mode and language instruction appear initially and when changed, while their
effects continue on subsequent steps. A mode change or target change occurs at
the midpoint with independent probability 0.35. The train/validation/test split
groups worlds by their dimensions, initial agent position and two marker
positions; changing only the goal or control mode cannot cross the split.

The teacher supplies one shortest-path action, with a fixed ordering to break
ties. Reported teacher agreement therefore differs from the rate of choosing
any optimal action. The actor reports count being on the current target at
each step, including remaining there; this is not an episode success rate.

The continued checkpoint uses the same retained 1,024 source episodes together
with 16 newly collected episodes. Only the retained source episodes contain
teacher actions and target-word labels. New experience contributes its executed
actions, observed image/audio outcomes and rewards, not the evaluator's hidden
world annotations. This run checks that collection and replay learning operate;
a single short cycle cannot establish a general improvement from online learning.

### Measured Results — 2026-09-10

The final test contains 128 unseen layouts and 768 recorded decisions. The
same histories are used before and after the experience cycle.

| Measurement | After 3,000 updates | After the experience cycle |
| --- | ---: | ---: |
| Teacher action agreement | 79.04% (607/768) | 79.17% (608/768) |
| Target-word exact match | 100.00% | 100.00% |

One extra matching decision does not establish an improvement from replaying
self-collected experience. The cycle changed 100 model state tensors, retained
the original training sources and included 16 new episodes without privileged
labels. An equal-compute continuation using only the old data was not run, so
these measurements also cannot attribute any change specifically to new data.

The final frozen model was evaluated again on those same histories:

| Input condition | Teacher action agreement | Target-word exact match |
| --- | ---: | ---: |
| All inputs and recurrent memory | 79.17% | 100.00% |
| Empty text throughout | 61.46% | 50.52% |
| Blank images throughout | 56.77% | 99.74% |
| Silent audio throughout | 53.52% | 100.00% |
| Reset memory before each observation | 63.28% | 61.46% |
| Best constant response on this test | 26.43% | 51.17% |

This supports dependence on language, image, sound and recurrent history in
the integrated policy. At the first observation, full and reset-memory
conditions agree exactly; on later observations, discarding earlier cues
reduces performance. These are short histories within the training task
family, not a demonstration of general conversation or long-term memory.

Forecast errors use the recorded executed action, with ordinary unweighted
MSE on the native source values:

| Forecast | Learned model | Simple reference |
| --- | ---: | ---: |
| Next RGB image | 0.06874 | 0.05842, copy the previous image |
| Next waveform | 0.02481 | 0.06736, predict silence |

Image prediction remains weaker than persistence and visually blurred. The
policy is learned from action labels and rewards; it selects actions directly
from memory rather than planning through the generated images. Successful
navigation therefore does not imply a reliable visual world model. Audio
prediction beats silence on the synthetic cues, without establishing speech
generation. Reward prediction MSE is 0.09135. The training report's image loss
weights changing pixels, so its values should not be compared directly with
the ordinary MSE table above.

The final actor executed 16 additional unseen-layout episodes of eight steps.
Its target word was correct on all 128 steps, teacher action agreement was
85.16%, and it occupied the current target on 57/128 steps. The
[offline replay](../models/multimodal-agent-20260910/replay.html) shows all 16
episodes in generation order, including failed actions and blurred forecasts.
The earlier eight-episode actor run used different seeds; their reward averages
are not a matched before/after comparison.

### Artifacts And Verification

Durable files are under `models/multimodal-agent-20260910/`:

- `pretrain/checkpoint.pt`: the fixed 3,000-update model.
- `cycle/round-000/learning/checkpoint.pt`: the continued model after 100 updates.
- [Test before](../models/multimodal-agent-20260910/test-before.json),
  [test after](../models/multimodal-agent-20260910/test-after.json) and
  [input diagnostics](../models/multimodal-agent-20260910/input-diagnostics.json):
  aggregate scores and individual episode/decision records.
- `before/`, `cycle/round-000/`, `after/`: native source episodes, explicit
  replay selections, actual actor decisions and predicted/observed PNG and WAV files.
- [Verification](../models/multimodal-agent-20260910/verification.json) and
  [manifest](../models/multimodal-agent-20260910/manifest.json): integrity checks,
  source identities, code provenance and artifact hashes.
- `runpod_timings.json` and `resource_summary.json`: measured runtime and resources.

The original source selection remains at
`data/multimodal-navigation-20260910/selection.json`. Its media hashes and all
copied actor sources were checked after retrieval. Test worlds were excluded
from the experience cycle, and displayed actor worlds were absent from training.
The suite passed 497 tests; desktop/mobile replay checks covered all 128 frames,
navigation, audio and absence of horizontal overflow or JavaScript errors.
Both disposable GPU pods used for this work were confirmed deleted.

Training and actor execution used commit `99bf8c8`. Later commits add the replay
viewer, input diagnostics and cumulative training-history checks; they do not
change the model or loss used by this run. This run retained its original replay
sources during continuation, so both checkpoints already record every training
world needed for the held-out checks.

## Commands

The CLI is `python -m intrep.problems.multimodal_agent.cli`. Its subcommands are:

| Command | Effect |
| --- | --- |
| `prepare` | Save native-media episodes and an explicit train/validation/test selection. |
| `train` | Jointly learn from selected episodes sampled by replay; optionally resume or initialize. |
| `evaluate` | Score a fixed checkpoint on a declared split. |
| `rollout` | Let the model act, save actual experiences, forecasts and a new replay selection. |
| `cycle` | Collect experiences and learn from them together with retained data, then repeat. |
| `infer` | Read text, an image, a WAV file, an action and feedback; return outputs and save session memory. |

An inference call can use any available combination of input modalities:

```sh
uv run python -m intrep.problems.multimodal_agent.cli infer \
  --checkpoint models/multimodal-agent-20260910/cycle/round-000/learning/checkpoint.pt \
  --text "Find the blue marker." \
  --image observation.png --audio cue.wav --output runs/agent-turn-1
uv run python -m intrep.problems.multimodal_agent.cli infer \
  --checkpoint models/multimodal-agent-20260910/cycle/round-000/learning/checkpoint.pt \
  --session runs/agent-turn-1/session.pt \
  --image next-observation.png --audio result.wav \
  --previous-action 3 --feedback 0.1 0 0 --output runs/agent-turn-2
```

The output contains generated text, action values and choice, image/audio
forecasts, predicted feedback and the updated memory snapshot. Keep snapshots
outside `runs/` when they must survive run cleanup. Input capability does not
imply a trained checkpoint understands every possible input distribution.

Saved actor traces can be reviewed without running a model in the browser:

```sh
uv run python scripts/render_multimodal_replay.py \
  --input models/multimodal-agent-20260910/after \
  --output models/multimodal-agent-20260910/replay.html
```
