# d256-h1024-heads8-l6-shogi

これは現在採用している将棋 policy/value checkpoint です。

`runs/shogi/policy-value-engine-analysis-1000games-runpod/best_checkpoint.pt`
から昇格しました。前回の smoke best checkpoint に対して MCTS8 20局で
20勝0敗0分だったため、今後の対局生成、評価、継続学習の基準として残します。

## メモ

- 構成: d256-h1024-heads8-l6
- 学習データ: `policy-value-engine-analysis-1000games`
- 学習は step 1750 で early stop
- best eval step: 750
- best eval loss: 3.0005
- eval accuracy: 0.2528
- eval value loss: 0.1115
- MCTS8 評価: 20勝0敗0分
- illegal move: 0
- SHA256: `904e2281f62aa0b2b3a212219d8ebfd75a6912d4c8c9968c49c3d398f83fd472`

この README は実験レジストリではなく、人間向けのメモです。後で継続学習や
差し替えをした場合は、自然言語で分かる範囲を更新します。
