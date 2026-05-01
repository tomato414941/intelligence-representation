# Experiment 06: Observation Stream

## 目的

自然言語入力だけでなく、Observation Streamを扱う最小実装を作る。

Experiment 05は将来の自然言語入力用に残し、先に入力の一般化を行う。
ここでは、Observationの `modality` に応じて、Claim、Event、StateTransitionを作る。

## 入力

### Text Observation

```text
modality: text
payload:
  subject: 田中
  predicate: has
  object: 本
```

期待:

```text
Observation: 1
Claim: 1
Belief: 1
```

### Action Result Observation

```text
modality: action_result
payload:
  event_type: place
  actor: 佐藤
  object: 本
  location: 図書館
  before:
    has: [佐藤, 本]
  after:
    located_at: [本, 図書館]
```

期待:

```text
Observation: 1
Event: 1
StateTransition: 1
```

## 成功条件

```text
Observationがmodalityを持つ
text observationからClaimを作れる
action_result observationからEventとStateTransitionを作れる
Claimだけを前提にしない
```
