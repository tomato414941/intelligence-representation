# Shogi Online Replay Metrics Structured Schema

Status: closed
Priority: low

## Problem

Online replay `metrics.json` contains useful data, but facts with different
responsibilities are mixed at the top level.

For example, checkpoint paths, replay sampling facts, generation summaries,
gate summaries, training settings, training metrics, and experience-store append
results live beside each other. This is workable for code that knows the schema,
but it makes ad hoc inspection and status documentation more error prone.

## Desired Shape

Do not add a human-authored `summary` block. A summary would encode a judgement
about which facts matter.

Instead, structure the record by responsibility:

- `checkpoint`
- `replay`
- `generation`
- `gate`
- `training`
- `experience_store`

The goal is not to duplicate derived summaries. It is to put each fact in its
natural place so readers do not need to infer meaning from a flat list of keys.

## Close Condition

- New online replay metrics are grouped by responsibility.
- Existing detailed metrics remain available.
- Status/update scripts do not need to know deep nested paths for common fields.
