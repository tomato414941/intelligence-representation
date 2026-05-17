# Shogi Online Replay Metrics Summary Schema

Status: open
Priority: low

## Problem

Online replay `metrics.json` contains useful data, but the most commonly read
training numbers are nested under `metrics`.

For example, `eval_loss`, `initial_eval_loss`, `eval_accuracy`, and
`best_eval_step` are not top-level fields. This is workable for code that knows
the schema, but it makes ad hoc inspection and status documentation more error
prone.

## Desired Shape

Keep the detailed training result nested, but expose a small top-level summary
for the fields humans and automation commonly compare:

- iteration index
- replay size
- generated sampled examples
- seed sampled examples
- training steps
- initial eval loss
- final eval loss
- eval accuracy
- best checkpoint path
- gate result, when present

The goal is not to duplicate the whole training result. It is to make the
durable run record easy to read and hard to misinterpret.

## Close Condition

- New online replay metrics include a compact top-level summary.
- Existing detailed metrics remain available.
- Status/update scripts do not need to know deep nested paths for common fields.
