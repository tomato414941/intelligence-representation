# Shogi Self-Play Max-Plies Draw Quality

Status: closed
Priority: medium

## Problem

The 2026-05-15 online replay run produced many self-play games that ended by the
`max_plies` cap:

- self-play games: 256
- `max_plies` draws: 109
- max plies: 320

These records are still valid generated experience, but a high cap-draw rate can
make the training signal harder to interpret. It may indicate weak play,
repetitive deterministic play, insufficient exploration, search settings that
avoid decisive lines, or simply that the model has not learned endgame
conversion.

## Desired Shape

Self-play generation should report enough outcome quality information to decide
whether the generated data is useful for training.

Useful facts include:

- cap-draw count and rate
- average plies
- terminal game count
- side win split
- search and sampling settings
- whether exploration noise or opening sampling was enabled

The project should decide whether cap-draw games are kept, downweighted,
filtered, or treated as normal records for the current training target.

## Close Condition

- Self-play summaries make the `max_plies` draw rate visible.
- The training data path has an explicit policy for cap-draw records.
- Online replay experiment notes can state how cap-draw records were handled.

## Resolution

Generation summaries now expose `max_plies_draw_count`,
`max_plies_draw_rate`, `game_over_count`, and `game_over_rate` as explicit
quality facts. Online Replay preserves the same fields when it combines source
summaries.

Cap-draw records are kept as generated experience. For winner-based value
targets, a `max_plies` draw has no winner, so value supervision is unknown and
masked out by the value loss; policy supervision remains usable. The project
does not filter or downweight cap-draw records by default.
