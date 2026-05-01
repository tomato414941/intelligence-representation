# Experiment 21: Temporal Conflict Prediction

## 目的

時間だけでは解けない競合を扱う。

Experiment 20では、同じsubject/predicateに対して新しいObservationが古いObservationをsupersedeした。
Experiment 21では、同じsubject/predicate/timestampで異なるobjectが観測された場合、recencyで片方を選ばずconflictとして保持する。

## 例

```text
t1:
  located_at(鍵, 箱)

t3:
  located_at(箱, 棚)

t3:
  located_at(箱, 机)

action:
  find(太郎, 鍵, unknown)

prediction:
  conflict
  candidates:
    located_at(鍵, 棚)
    located_at(鍵, 机)
```

## recencyとの違い

```text
t2: located_at(箱, 棚)
t3: located_at(箱, 机)
  -> resolved by recency

t3: located_at(箱, 棚)
t3: located_at(箱, 机)
  -> conflict
```

## 評価

```text
prediction_state:
  unsupported | resolved | conflict

candidate_facts:
  conflict時の候補

conflict_observation_ids:
  競合しているObservation

superseded_observation_ids:
  recencyで置き換えられたObservation
```

## 成功条件

```text
同時刻の競合では片方をlatestとして選ばない
prediction_stateがconflictになる
候補factを複数保持する
競合ObservationのIDを保持する
時刻が違う場合はrecencyでresolvedにできる
```
