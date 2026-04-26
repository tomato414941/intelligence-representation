# Experiment 22: Reliability-Weighted Prediction

## 目的

同時刻の競合を、source reliabilityで解決できる場合とできない場合に分ける。

Experiment 21では、同じsubject/predicate/timestampで異なるobjectがある場合、常にconflictとして扱った。
Experiment 22では、信頼度差が十分大きければ高信頼ソースを採用し、低信頼側をcounterevidenceとして残す。

## 例

```text
t1 sensor:
  located_at(鍵, 箱)

t3 sensor:
  located_at(箱, 棚)
  reliability = 0.95

t3 user_guess:
  located_at(箱, 机)
  reliability = 0.40

prediction:
  state = resolved_with_uncertainty
  fact = located_at(鍵, 棚)
  confidence = 0.95
  counter_candidates = [located_at(鍵, 机)]
```

## 比較

```text
信頼度差が大きい:
  resolved_with_uncertainty

信頼度差が小さい:
  conflict
```

## 評価

```text
prediction_state:
  unsupported | resolved | resolved_with_uncertainty | conflict

confidence:
  採用したObservationのreliability

counter_candidates:
  採用しなかった候補

provenance_observation_ids:
  採用した根拠

counterevidence_observation_ids:
  採用しなかった根拠
```

## 成功条件

```text
信頼度差が大きい同時刻競合をresolved_with_uncertaintyにできる
低信頼候補をcounter_candidateとして残す
信頼度差が小さい場合はconflictのまま保持する
provenanceとcounterevidenceを保持する
```
