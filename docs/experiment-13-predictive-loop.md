# Experiment 13: Predictive Loop

## 目的

世界モデルの中心ループを最小実装する。

```text
Observation
  ↓
State
  ↓
Prediction
  ↓
New Observation
  ↓
Prediction Error
  ↓
State Update
```

この実験では、予測モデルは手設計ルールでよい。
目的は賢い予測器を作ることではなく、予測、観測、誤差、更新の流れを明示することである。

## 例

初期状態:

```text
has(佐藤, 本)
```

action:

```text
place(佐藤, 本, 図書館)
```

prediction:

```text
located_at(本, 図書館)
```

new observation:

```text
located_at(本, 図書館)
```

prediction_error:

```text
none
```

## 不一致例

prediction:

```text
located_at(本, 図書館)
```

new observation:

```text
located_at(本, 机)
```

prediction_error:

```text
mismatch
```

## 成功条件

```text
actionからpredictionを作れる
new observationとpredictionを比較できる
一致ならconfirmedにできる
不一致ならprediction_errorを記録できる
errorからstateを更新できる
```
