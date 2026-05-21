# Evaluation Roster

This roster lists model entries that should be kept visible as comparison
targets. Entry names are human-facing labels. Exact identity belongs in specs,
manifests, hashes, checkpoints, and metrics.

## Shogi Model Entries

| Entry | Input intent | Status |
| --- | --- | --- |
| `shogi-policy-plane-minimal-single-global` | Board pieces + one global summary element for state, side, move count, and hands. | implemented |
| `shogi-policy-plane-minimal-split-global` | Board pieces + separate global elements for state, side, move count, and hands. | implemented |
| `shogi-policy-plane-alpha-zero-like-no-history` | Current-state piece planes + split global side, move count, and hands; no history. | implemented |
| `shogi-policy-plane-dlshogi-like` | AlphaZero-like current state + check and attack features; no history. | implemented |
| `shogi-policy-plane-rich` | Rich features: piece, line, pair, drop, and tactical hints. | implemented |
