# Evaluation Roster

This roster lists model entries that should be kept visible as comparison
targets. Entry names are labels for humans. Checkpoint locations, training
results, match results, and runtime measurements belong in status or measurement
documents.

## Shogi Model Entries

| Entry | Input intent |
| --- | --- |
| `shogi-action-plane-policy-output-minimal-single-global` | Board pieces + one global summary element for state, side, move count, and hands. |
| `shogi-action-plane-policy-output-minimal-split-global` | Board pieces + separate global elements for state, side, move count, and hands. |
| `shogi-action-plane-policy-output-alpha-zero-like-no-history` | Current-state piece planes + split global side, move count, and hands; no history. |
| `shogi-action-plane-policy-output-dlshogi-like-no-entering-king` | dlshogi-style current-position piece, hand, check, and attack features; no entering-king features. |
