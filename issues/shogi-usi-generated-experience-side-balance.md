# Shogi USI Generated Experience Side Balance

Status: open
Priority: high

## Problem

The current shogi USI generated experience source plays a fixed-side matchup:

- black: checkpoint
- white: USI engine

This makes USI-engine experience structurally biased by side. If the USI engine wins most or all games, the generated data also entangles opponent strength, side, and outcome.

## Desired Shape

USI generated experience should support side-balanced generation:

- checkpoint as black vs USI as white
- USI as black vs checkpoint as white

The generated game records should keep normal actor metadata so downstream experience store, replay, and training code can consume both sides without special cases.

## Close Condition

- USI generated experience can generate both side assignments in one online replay cycle.
- The generation summary records the side split.
- Tests cover both `checkpoint:usi_engine` and `usi_engine:checkpoint` generated records.
