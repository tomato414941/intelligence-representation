# Shogi Model Entry And Cache Artifact Naming

Status: open. Priority: medium.

## Issue

Shogi naming currently mixes three different identities:

- assembly spec ids used by code
- model entry names used by humans in the evaluation roster
- tensor cache artifact names used in local/R2 storage

The code-side assembly spec ids are mostly explicit, for example:

- `shogi_policy_value_minimal_single_global_position_transformer_action_plane_policy`
- `shogi_policy_value_alpha_zero_like_position_transformer_action_plane_policy`
- `shogi_policy_value_dlshogi_like_no_entering_king_position_transformer_action_plane_policy`

These names say input representation, core, and output representation clearly.

The human-facing model entry names are less clear:

- `shogi-action-plane-policy-output-minimal-single-global`
- `shogi-action-plane-policy-output-minimal-split-global`
- `shogi-action-plane-policy-output-alpha-zero-like-no-history`
- `shogi-action-plane-policy-output-dlshogi-like-no-entering-king`

They put the output side first even though the current comparison axis is the
shogi position input representation. This makes the roster harder to read.

The cache artifact names are also inconsistent. Recent R2 prefixes use names
like:

- `shogi-minimal-single-global-action-plane`
- `shogi-alpha-zero-like-no-history-action-plane`
- `shogi-dlshogi-like-no-entering-king-action-plane`

These names are understandable, but they do not say that the first part is a
position input representation.

## Desired Naming Shape

Keep machine ids and human labels separate.

Assembly spec ids should remain precise and code-facing:

- domain/problem: `shogi_policy_value`
- input: `minimal_single_global_position`, `alpha_zero_like_position`, etc.
- core: `transformer`
- output: `action_plane_policy`

Model entry names should be concise human labels for evaluation:

- `shogi-minimal-single-global-position-action-plane`
- `shogi-minimal-split-global-position-action-plane`
- `shogi-alpha-zero-like-no-history-position-action-plane`
- `shogi-dlshogi-like-no-entering-king-position-action-plane`

Tensor cache artifact names should either match the model entry name or be
derived from it. Dataset identity should stay in the parent path, for example:

```text
r2://intrep/shogi/tensor-caches/qhapaq-full/shogi-minimal-single-global-position-action-plane
```

This keeps the artifact name focused on representation shape and keeps source
data identity in the directory hierarchy.

## Current Assessment

The assembly spec ids do not need immediate renaming.

The evaluation roster names should be renamed first because they are
human-facing and currently point attention at the wrong axis.

The generated cache artifact names should be renamed or re-released under the
same naming policy before they become long-lived references in docs, scripts,
or training commands.

The default `action-plane-policy` cache name is too generic for long-lived
artifacts. It is acceptable only as a local test fixture or short-lived
temporary cache directory.

## Acceptance Criteria

This issue can close when:

- `docs/shogi/evaluation-roster.md` uses position-aware model entry names
- RunPod/R2 cache release names use the same model-entry-style naming policy
- scripts do not default long-lived shogi cache artifacts to `action-plane-policy`
- docs and examples use the same names consistently
- existing assembly spec ids are left alone unless a separate issue identifies a
  concrete ambiguity in code-facing names

## Non-Goals

- rename all internal Python classes or modules
- change model behavior
- redesign tensor cache layout
- introduce aliases for old names
- define a generic project-wide artifact naming framework
