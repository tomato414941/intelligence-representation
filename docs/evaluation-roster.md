# Evaluation Roster

This roster lists model entries that should be kept visible as comparison
targets. Entry names are human-facing labels. Exact identity belongs in specs,
manifests, hashes, checkpoints, and metrics.

## Shogi Policy-Plane Entries

Keep the output, core, training data, and training recipe fixed when comparing
these entries. The intended comparison axis is the shogi position input
representation.

| Entry | Input intent | Status |
| --- | --- | --- |
| `shogi-policy-plane` | Rich position features: square, piece, line, pair relations, drop shadow, counterfactual, and capture-flow style features. | implemented |
| `shogi-policy-plane-alpha-zero-like-no-history` | Current-position AlphaZero-style state description: piece placement, hand counts, side-to-move, and move count; no attack, line, pair, drop-shadow, counterfactual, capture-flow, or position-history features. | wanted |
| `shogi-policy-plane-dlshogi-like` | dlshogi/PGX-style current-position features: piece placement, piece-type attacks, attack count thresholds, hand count thresholds, and in-check; no line, pair, drop-shadow, counterfactual, or capture-flow features. | wanted |

