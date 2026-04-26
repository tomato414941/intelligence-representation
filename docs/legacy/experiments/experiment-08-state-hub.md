# Experiment 08: State Hub

## 目的

`Observation -> StateUpdate -> State` の最小統合を作る。

これまでの実験では、Semantic Memory、Observation Stream、WorldStateが別々だった。
この実験では、Observationを受け取り、modalityに応じて複数種類のStateを更新する。

## 最小構成

```text
ObservationStore:
  すべての入力を保存する

WorldState:
  action_resultから得た状態変化を反映する

BeliefState:
  text由来のClaimを統合する

ConflictState:
  競合するBeliefを保持する

MemoryProvenanceState:
  ObservationがどのStateを作ったかを記録する
```

## Text Observation

入力:

```text
modality: text
payload:
  subject: 田中
  predicate: has
  object: 本
```

更新:

```text
ObservationStoreに保存
BeliefStateにbeliefを追加またはmerge
MemoryProvenanceStateにobservation -> beliefを記録
```

## Action Result Observation

入力:

```text
modality: action_result
payload:
  before:
    has: [佐藤, 本]
  after:
    located_at: [本, 図書館]
```

更新:

```text
ObservationStoreに保存
WorldStateを更新
MemoryProvenanceStateにobservation -> factを記録
```

## Conflict

入力:

```text
田中が本を持っている
佐藤が本を持っている
```

更新:

```text
BeliefStateに2つのbelief
ConflictStateにconflict
```

## 成功条件

```text
Observationが保存される
text observationがBeliefStateを更新する
action_result observationがWorldStateを更新する
矛盾するtext observationがConflictStateを更新する
Provenanceでobservationから更新先をたどれる
```
