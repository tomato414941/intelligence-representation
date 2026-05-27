# Shogi Playing Strength

This document records shogi playing-strength checks. Runtime measurements live
in the throughput and inference-performance docs.

`runs/` is disposable. Match results that should survive must be summarized
here.

## Detailed Results

| Case | Date | Entry | Opponent | Opponent setting | Games | Search setting | Start positions | Result | Illegal moves | Avg plies | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| `alpha_zero_like_vs_minimal_single_global_mcts128_b64_g16` | 2026-05-26 | shogi-action-plane-policy-output-alpha-zero-like-no-history vs shogi-action-plane-policy-output-minimal-single-global | none | entry comparison | 16 | MCTS128, NN leaf eval batch limit 64 | 8 seeded random-legal openings with both side assignments; seed 20260525; opening plies 8 | alpha-zero-like 2, single-global 13, draws 1 | 0 | 144.25 | Alpha-zero-like side split: black 1-6, white 1-7. |
| `alpha_zero_like_vs_minimal_split_global_mcts128_b64_g16` | 2026-05-26 | shogi-action-plane-policy-output-alpha-zero-like-no-history vs shogi-action-plane-policy-output-minimal-split-global | none | entry comparison | 16 | MCTS128, NN leaf eval batch limit 64 | 8 seeded random-legal openings with both side assignments; seed 20260525; opening plies 8 | alpha-zero-like 5, split-global 11, draws 0 | 0 | 167.0625 | Alpha-zero-like side split: black 2-6, white 3-5. |
| `alpha_zero_like_vs_suisho5_n1_mcts128_b64_g16` | 2026-05-26 | shogi-action-plane-policy-output-alpha-zero-like-no-history | Suisho5 | `go nodes 1` | 16 | MCTS128, NN leaf eval batch limit 64 | alternating sides | 2-14-0 | 0 | 109.5 | Side split: black 1-7, white 1-7. |
| `single_global_vs_suisho5_n1000_mcts128_b64_g16` | 2026-05-25 | shogi-action-plane-policy-output-minimal-single-global | Suisho5 | `go nodes 1000` | 16 | MCTS128, NN leaf eval batch limit 64 | alternating sides | 4-12-0 | 0 | 107.375 | Side split: black 1-7, white 3-5. |
| `split_global_vs_single_global_mcts128_b64_g16` | 2026-05-25 | shogi-action-plane-policy-output-minimal-split-global vs shogi-action-plane-policy-output-minimal-single-global | none | entry comparison | 16 | MCTS128, NN leaf eval batch limit 64 | 8 seeded random-legal openings with both side assignments; seed 20260525; opening plies 8 | split-global 8, single-global 7, draws 1 | 0 | 192.0625 | Split-global side split: black 5-3, white 3-4. |
| `split_global_vs_suisho5_n1_mcts128_b64_g16` | 2026-05-24 | shogi-action-plane-policy-output-minimal-split-global | Suisho5 | `go nodes 1` | 16 | MCTS128, NN leaf eval batch limit 64 | alternating sides | 9-7-0 | 0 | 180.4375 | Side split: black 7-1, white 2-6. |
| `split_global_vs_suisho5_n1000_mcts128_b64_g16` | 2026-05-24 | shogi-action-plane-policy-output-minimal-split-global | Suisho5 | `go nodes 1000` | 16 | MCTS128, NN leaf eval batch limit 64 | alternating sides | 7-9-0 | 0 | 141.1875 | Side split: black 3-5, white 4-4. |
