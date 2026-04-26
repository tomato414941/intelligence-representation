# Current Results

## 位置づけ

この文書は、現在の主線であるExperiment 17-23の結果を短くまとめる。

結論を先に言うと、ここまでで確認できたのは、学習する世界モデルではない。
確認できたのは、toy symbolic environment上で、memory、retrieval、multi-hop、time、conflict、reliabilityが予測結果に影響することだけである。

## Results

| Experiment | Question | Result |
| --- | --- | --- |
| 17 Observation-assisted prediction | retrieveされた観測は予測を改善するか | `retrieved_memory` が `accuracy=1.00`、`no_memory` と `recent_memory` は `0.50` |
| 18 Multi-hop observation prediction | 直接検索だけで足りるか | `direct_memory=0.00`、`multi_hop_memory=1.00` |
| 19 Ambiguous multi-hop prediction | 一意に解けないmulti-hopを潰さず扱えるか | `ambiguous_rate=1.00`、候補集合として正解 |
| 20 Temporal multi-hop prediction | 時間差で現在状態を更新できるか | 最新観測を使い `accuracy=1.00`、古い観測は `superseded` として保持 |
| 21 Temporal conflict prediction | 同時刻の競合をrecencyで誤解決しないか | `conflict_rate=1.00`、候補集合として保持 |
| 22 Reliability-weighted prediction | 信頼度差で競合を一部解決できるか | `resolved_with_uncertainty_rate=0.50`、`conflict_rate=0.50` |
| 23 Learned transition predictor | 環境生成データから予測規則を学習できるか | `learned_accuracy=1.00`、`rule_accuracy=0.17` |

## What This Shows

ここまでで示したこと:

```text
retrievalは、直近memoryより有効な場合がある
直接検索だけではmulti-hop予測に足りない
一意に解けない状態は候補集合として保持できる
時間差がある更新はrecencyで扱える
同時刻の競合はrecencyでは解けない
信頼度差が十分大きい場合だけ、競合をresolved_with_uncertaintyにできる
小さな環境生成データでは、学習baselineが手書きruleを上回る
```

## What This Does Not Show

未対応の批判はまだ残っている。

```text
latent stateを持っていない
Transformer predictorを使っていない
予測誤差による表現更新を学習していない
大量データ、ノイズ、部分観測、分布外汎化を扱っていない
行動選択やplanningを改善していない
実験はまだ手書きFact/Actionと小さな環境に依存している
```

したがって、現在の到達点は「世界モデルができた」ではない。

より正確には、

```text
世界モデル評価へ向かうための、
小さな予測タスクと評価軸を作った段階
```

である。

## Next Pressure

次に増やすべきものは、新しいState分類ではない。

次に必要なのは、次のどちらかである。

```text
1. 生成データを増やし、train/testだけでなくheld-out actionやheld-out objectで評価する
2. Frequency baselineを、より一般化できるsequence / vector predictorに置き換える
```

どちらの場合も、成功条件は次である。

```text
memory / retrieval / context building が prediction error を下げるか
context size と accuracy のトレードオフが見えるか
未知ケースに汎化するか
```
