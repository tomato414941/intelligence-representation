# Experiment 20: Temporal Multi-Hop Prediction

## 目的

時間つきObservationを使い、現在状態と履歴を分けて扱う。

Experiment 19では、複数候補をambiguousとして保持した。
Experiment 20では、同じsubject/predicateに対する観測が時間差で入った場合、最新観測を現在状態として使い、古い観測は消さずにsupersededとして残す。

## 例

```text
t1:
  located_at(鍵, 箱)

t2:
  located_at(箱, 棚)

t3:
  located_at(箱, 机)

action at t4:
  find(太郎, 鍵, unknown)

prediction:
  located_at(鍵, 机)
```

## 重要な点

古いObservationは削除しない。

```text
current prediction:
  located_at(鍵, 机)

provenance:
  obs_1: located_at(鍵, 箱)
  obs_3: located_at(箱, 机)

superseded:
  obs_2: located_at(箱, 棚)
```

## 評価

```text
predicted_fact
expected_fact
provenance_observation_ids
superseded_observation_ids
accuracy
```

## 成功条件

```text
同じsubject/predicateでは最新Observationを使う
multi-hopの途中でも最新Observationを使う
古いObservationをsupersededとして保持する
元Observationは削除しない
```
