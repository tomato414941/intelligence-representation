# Experiment 17: Observation-Assisted Prediction

## 目的

Observation Memoryが世界モデル予測に役立つかを、最小の比較実験として測る。

中心仮説は次である。

```text
過去観測を適切にretrieveしてPredictorへ渡すと、
memoryなしより次状態予測が改善する。
```

## 比較条件

```text
no_memory:
  state_before + action だけで予測する

recent_memory:
  直近k件のObservationをcontextとして使う

retrieved_memory:
  actionに関連するObservationだけretrieveして使う
```

## 評価

評価対象は既存のActionConditionedExampleである。

```text
state_before + action -> expected_observation
```

記録するものは次である。

```text
accuracy
context_size
retrieved_observation_ids
case別の正誤
```

## 意味

これはTransformerPredictorではない。

まだ学習器は入れず、まず次を確認する。

```text
全部の記憶を渡す必要はない
直近だけでは足りない場合がある
retrieveされた観測が予測に効く場合がある
```

この実験により、外部メモリは主役ではなく、予測ループに渡す材料を選ぶ補助装置として評価される。

## 成功条件

```text
no_memoryよりretrieved_memoryのaccuracyが高い
retrieved_memoryが必要Observationを選ぶ
recent_memoryが古い重要Observationを見落とすケースがある
context_sizeを記録できる
```
