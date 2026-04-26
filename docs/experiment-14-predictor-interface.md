# Experiment 14: Predictor Interface

## 目的

Predictive Loopの予測器を差し替え可能にする。

Experiment 13では、`place` actionの予測を `PredictiveWorld` の中にハードコードしていた。
この実験では、予測モデルを外に出し、後でLLM、learned model、simulatorに差し替えられるようにする。

## インターフェース

```text
Predictor:
  predict(state, action) -> Fact | None
```

## 最小実装

```text
RuleBasedPredictor:
  place(actor, object, target)
    -> located_at(object, target)

PredictiveWorld:
  predictorを注入して使う
```

手設計ルールは本体ではなく、baseline adapterとして扱う。

## 成功条件

```text
RuleBasedPredictorがplace actionを予測できる
PredictiveWorldが注入されたPredictorを使う
unsupported actionは予測なしとして扱える
prediction mismatchを記録できる
```
