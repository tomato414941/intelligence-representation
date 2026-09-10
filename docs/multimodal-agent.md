# Multimodal Predictive Agent

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
  --checkpoint models/multimodal-agent/checkpoint.pt \
  --text "Find the blue marker." \
  --image observation.png --audio cue.wav --output runs/agent-turn-1
uv run python -m intrep.problems.multimodal_agent.cli infer \
  --checkpoint models/multimodal-agent/checkpoint.pt \
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
  --input models/multimodal-agent-results/after \
  --output models/multimodal-agent-results/replay.html
```
