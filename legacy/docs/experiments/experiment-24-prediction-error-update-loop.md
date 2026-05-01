# Experiment 24: Prediction Error Update Loop

## 目的

予測誤差によってpredictorを更新する最小ループを作る。

Experiment 23では、環境生成データから学習可能なpredictor baselineを作った。
Experiment 24では、未知ケースで失敗したあと、その観測をtraining memoryへ追加し、再fitによって同じケースを予測できるようにする。

## ループ

```text
predict
  ↓
observe
  ↓
compare
  ↓
record prediction error
  ↓
append error case to training memory
  ↓
refit
  ↓
predict again
```

## 評価ケース

初期training memoryには `鍵`, `箱`, `本` のケースだけがある。

未知ケース:

```text
state_before:
  located_at(財布, ケース)
  located_at(ケース, 引き出し)

action:
  find(太郎, 財布, unknown)

observed:
  located_at(財布, 引き出し)
```

初回はunsupportedになる。
観測済みケースをtraining memoryへ追加して再fitすると、同じケースを予測できる。

## 記録する値

```text
prediction_error_type
predicted_before
predicted_after
observed
before_correct
after_correct
training_size_before
training_size_after
```

## 限界

これはモデル重みのオンライン学習ではない。

更新しているのは `FrequencyTransitionPredictor` のtraining examplesである。
ただし、`prediction -> error -> update -> improved prediction` の最小ループとして、中心批判に直接対応する。
