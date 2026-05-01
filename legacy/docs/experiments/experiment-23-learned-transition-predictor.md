# Experiment 23: Learned Transition Predictor

## 目的

手書きケースだけでなく、小さな状態遷移環境からaction-conditioned dataを生成し、学習可能なpredictor baselineを評価する。

これは本物のlatent world modelではない。
ただし、手書きruleだけのpredictorから、訓練データから予測を獲得するbaselineへ進む。

## 構成

```text
MiniTransitionEnvironment:
  object -> location の状態を持つ

Actions:
  place
  move_container
  find

Generated Example:
  state_before
  action
  expected_observation
  expected_state_after
```

## Predictor

```text
FrequencyTransitionPredictor:
  train examplesから、
  action patternに対する最多expected_observationを学習する

Unknown pattern:
  unsupported
```

比較対象は既存の `RuleBasedPredictor` である。

## 評価

```text
train / test split
rule_accuracy
rule_unsupported_rate
learned_accuracy
learned_unsupported_rate
```

## 成功条件

```text
環境から複数のActionConditionedExampleを生成できる
train/test splitを持つ
learned predictorがtrain examplesから予測できる
unknown patternはunsupportedになる
同じ評価ハーネスでRuleBasedPredictorと比較できる
```

## 限界

```text
latent stateはない
Transformer predictorではない
sequence modelではない
予測誤差から表現を更新していない
まだtoy symbolic environmentである
```
