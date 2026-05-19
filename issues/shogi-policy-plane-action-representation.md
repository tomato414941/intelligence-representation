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
- Added `PolicyPlaneValueTensorSample` and policy-plane tensorization from
  `ShogiMovePolicyValueExample`.
- Policy-plane tensorization maps chosen-move targets and weighted move-policy
  targets into the fixed action space, and builds the legal action mask.
- Added `ShogiPolicyPlaneHead` as a standalone fixed-action policy head. It is
  not yet wired into a full model variant or training loop.

Remaining work:

- Decide whether policy-plane output should replace candidate scoring or coexist
  behind a separate model.
- Add a policy-plane model variant and training path.
- Connect MCTS priors to policy-plane outputs.

## Policy Output Space Boundary

`ShogiMovePolicyValueExample` is a Training Example in the glossary sense: it is a
meaning-level unit that records a position, legal moves, a move target or move
target distribution, and an optional value target. It is not yet tied to a
candidate-move-policy output space or a fixed policy-plane output space.

That makes it reasonable as the common source for both policy outputs:

```text
ShogiMovePolicyValueExample
  -> candidate-move-policy sample
  -> policy-plane sample
```

The name is intentionally move-specific because the policy target is a move
target. It should stay shared only while candidate-move policy and policy plane
consume the same move-policy/value Training Example.

`CandidateMovePolicyValueTensorSample` is explicitly the existing
candidate-move-policy/value runtime sample:

- candidate move features
- candidate-length policy target tensor
- chosen-move index within `legal_moves`
- padded candidate mask at dataset collation time

Do not add policy-plane fields directly to that sample unless the sample schema
is first split or renamed. The cleaner migration path is:

```text
CandidateMovePolicyValueTensorSample
PolicyPlaneValueTensorSample
```

Both may share position input tensors and value targets, but they should own
separate policy target tensors, masks, losses, metrics, tensor-cache schemas,
and model heads.

The serialized data-selection source kind remains
`shogi_policy_value_examples_jsonl` for now. Renaming that artifact schema should
be a separate migration decision because existing bundles and data-selection
files already use it.
