# Experiment 15: Predictor Evaluation

## 目的

Predictorを差し替えて比較できる評価ハーネスを作る。

Experiment 14では、Predictorを注入可能にした。
次は、RuleBasedPredictor、LLM Predictor、learned predictor、simulator predictorを同じ評価ケースで比較できるようにする。

## 評価ケース

```text
PredictionCase:
  name
  initial_state
  action
  expected_fact
```

`expected_fact` がない場合は、unsupportedが期待値である。

## 指標

```text
accuracy:
  expected_fact と prediction が一致した割合

unsupported_rate:
  predictorが予測不能を返した割合
```

## 例

```text
initial_state:
  has(佐藤, 本)

action:
  place(佐藤, 本, 図書館)

expected:
  located_at(本, 図書館)
```

unsupported例:

```text
action:
  throw(佐藤, 本, 床)

expected:
  unsupported
```

## 成功条件

```text
Predictorを差し替えられる
accuracyを計算できる
unsupported_rateを計算できる
RuleBasedPredictorの得意・不得意が評価で見える
```
