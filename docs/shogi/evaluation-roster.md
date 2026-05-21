# Evaluation Roster

This roster lists model entries that should be kept visible as comparison
targets. Entry names are labels for humans. Exact run identity is recorded in specs,
manifests, checkpoints, and metrics.

## Shogi Model Entries

| Entry | Input intent | Status |
| --- | --- | --- |
| `shogi-action-plane-policy-output-minimal-single-global` | Board pieces + one global summary element for state, side, move count, and hands. | implemented |
| `shogi-action-plane-policy-output-minimal-split-global` | Board pieces + separate global elements for state, side, move count, and hands. | implemented |
| `shogi-action-plane-policy-output-alpha-zero-like-no-history` | Current-state piece planes + split global side, move count, and hands; no history. | implemented |
| `shogi-action-plane-policy-output-dlshogi-like` | dlshogi-style current-position piece, hand, check, and attack features. | implemented |
