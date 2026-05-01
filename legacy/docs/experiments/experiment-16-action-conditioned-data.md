# Experiment 16: Action-Conditioned Data

## 目的

Predictorを学習・比較するための最小データ形式を作る。

Experiment 15では、Predictorの評価ハーネスを作った。
次は、RuleBasedPredictorだけでなく、将来のTransformerPredictorやlearned predictorにも渡せる共通データを用意する。

## 最小形式

```text
ActionConditionedExample:
  id
  state_before
  action
  expected_observation
  expected_state_after
  source
```

中心はこれである。

```text
state_before + action -> expected_observation / expected_state_after
```

## 例

```json
{
  "id": "place_book_library",
  "state_before": [
    {"subject": "佐藤", "predicate": "has", "object": "本"}
  ],
  "action": {
    "type": "place",
    "actor": "佐藤",
    "object": "本",
    "target": "図書館"
  },
  "expected_observation": {
    "subject": "本",
    "predicate": "located_at",
    "object": "図書館"
  },
  "expected_state_after": [
    {"subject": "佐藤", "predicate": "has", "object": "本"},
    {"subject": "本", "predicate": "located_at", "object": "図書館"}
  ],
  "source": "manual"
}
```

## 位置づけ

これはontologyではない。

人間が細かい意味構造を固定するのではなく、予測器に渡せる観測・行動・期待結果の薄いデータ形式である。

```text
Action-conditioned data
  ↓
RuleBasedPredictor / TransformerPredictor / learned predictor
  ↓
prediction
  ↓
evaluation
```

## 成功条件

```text
JSONLとして保存できる
JSONLから復元できる
PredictionCaseへ変換できる
既存のPredictor評価ハーネスに渡せる
```
