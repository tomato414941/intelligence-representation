# Shogi Policy Plane Action Representation

Status: open
Priority: medium

## Problem

The current shogi policy head scores a variable list of legal candidate moves.
AlphaZero-style and dlshogi-style shogi systems instead use a fixed policy
plane action representation, then map legal moves into that fixed action space.

Candidate scoring is simpler and already works, but policy planes may make the
policy output more spatially structured and easier to connect to MCTS priors.

## Desired Shape

Investigate a shogi policy-plane representation as an output-side redesign:

- define the fixed action space
- define USI move to action-index mapping and inverse mapping
- define legal-move mask construction
- define policy-target tensors for chosen-move and MCTS-visit-count targets
- define how MCTS reads priors from policy planes
- decide whether candidate scoring and policy planes should coexist

This is not an input-feature change. It should not be implemented in the same
workstream as attack-map input features, otherwise strength and performance
effects will be hard to attribute.

## Cost And Risk

This will affect examples, tensor caches, policy targets, model heads, evaluation,
and MCTS integration. It is a larger change than adding attack input features.

## Close Condition

- The policy-plane action space is specified.
- A migration path from candidate scoring is chosen.
- Implementation work is split into concrete follow-up tasks.

## Progress

2026-05-19:

- Added a shogi policy-plane action-space module under
  `intrep.worlds.shogi.policy_plane`.
- The initial fixed action space is `81 * 43`:
  - 8 short directions
  - 8 short promotion directions
  - 8 long directions
  - 8 long promotion directions
  - 2 knight directions
  - 2 knight promotion directions
  - 7 drops
- Actions are side-to-move-relative and indexed as
  `relative_to_square * 43 + move_type`.
- Tests cover fixed action size, normal moves, promotion, drops, long moves,
  knight moves, side-to-move-relative indexing, legal masks, and legal move
  round-tripping through an action index.

Remaining work:

- Decide whether policy-plane output should replace candidate scoring or coexist
  behind a separate model.
- Add policy-plane target tensors.
- Add a policy-plane model head.
- Connect MCTS priors to policy-plane outputs.
