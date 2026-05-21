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
| `shogi-policy-plane-minimal-single-global` | Board pieces + one global summary element for state, side, move count, and hands. | implemented |
| `shogi-policy-plane-minimal-split-global` | Board pieces + separate global elements for state, side, move count, and hands. | implemented |
| `shogi-policy-plane-alpha-zero-like-no-history` | AlphaZero-style current state; no history. | wanted |
| `shogi-policy-plane-dlshogi-like` | dlshogi-style state plus attack features. | wanted |
| `shogi-policy-plane-rich` | Rich features: piece, line, pair, drop, and tactical hints. | implemented |
