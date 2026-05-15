# Shogi USI Generated Experience Side Balance

Status: closed
Priority: high

## Problem

The previous shogi USI generated experience source played a fixed-side matchup:

- black: checkpoint
- white: USI engine

This made USI-engine experience structurally biased by side. If the USI engine won most or all games, the generated data also entangled opponent strength, side, and outcome.

## Desired Shape

USI generated experience should support side-balanced generation:

- checkpoint as black vs USI engine as white
- USI engine as black vs checkpoint as white

The generated game records should keep normal actor metadata so downstream experience store, replay, and training code can consume both sides without special cases.

## Resolution

2026-05-15:

- Generated experience now uses explicit generated player specs instead of `opponent="usi"`.
- `ShogiGeneratedExperienceSource` now owns `black_player` and `white_player`.
- USI is represented as `kind="usi_engine"` in the generated player spec, not as the whole player/opponent concept.
- `run_shogi_generated_games` builds both black and white player CLI arguments from those specs.
- Online replay CLI supports explicit source forms:
  - `checkpoint-self:GAMES`
  - `checkpoint-black-vs-usi:GAMES`
  - `usi-black-vs-checkpoint:GAMES`
  - `checkpoint-vs-usi-balanced:GAMES`
- `checkpoint-vs-usi-balanced:GAMES` expands into both side assignments in one online replay cycle.
- Generation summaries record `black_player` and `white_player` for each source.
- Tests cover both checkpoint-black vs USI-white and USI-black vs checkpoint-white generated records.

## Close Condition

- USI generated experience can generate both side assignments in one online replay cycle.
- The generation summary records the side split.
- Tests cover both `checkpoint:usi_engine` and `usi_engine:checkpoint` generated records.
