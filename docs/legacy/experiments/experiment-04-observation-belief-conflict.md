# Experiment 04: Observation, Belief, Conflict

## 目的

Raw Observation、Semantic Unit、Belief、Conflict、Update Logを分ける。

Experiment 03までは、Claimを直接状態に追加し、矛盾候補をClaim上の `conflicts_with` として保持した。
Experiment 04では、矛盾を第一級オブジェクトとして扱い、観測と信念を分離する。

## 最小ストア

```text
Observation Store:
  生入力を保存する。immutable。

Semantic Unit Store:
  抽出されたClaimやRelationを保存する。

Belief Store:
  現在採用している理解や仮説を保存する。mutable。

Conflict Store:
  矛盾や競合仮説を保存する。

Update Log:
  状態変更履歴を保存する。
```

## 基本方針

```text
Raw observations は消さない
Derived beliefs は更新可能にする
Conflicts は第一級オブジェクトとして保存する
Belief には scope, evidence, confidence, time を持たせる
Transformerには全部を渡さず、必要な局所状態だけ渡す
```

## 更新型

```text
add:
  新しいClaimを追加する

merge:
  類似Claimを統合し、supporting_observationsを増やす

refine:
  既存Claimをより正確なClaimで更新する

contradict:
  既存Claimと衝突するClaimを登録し、Conflictを作る

override:
  新しいClaimが古いClaimを置き換える

retire:
  Claimを非アクティブ化する

split:
  Claimを文脈やスコープごとに分ける
```

## 処理の流れ

```text
新しい入力
  ↓
Observationとして保存
  ↓
Claim / Relationを抽出
  ↓
既存stateと照合
  ↓
重複ならmerge
  ↓
補強ならsupport追加
  ↓
修正ならrefine
  ↓
矛盾ならconflict登録
  ↓
必要ならbelief更新
```

## 成功条件

```text
観測をimmutableに保存できる
観測から派生したBeliefを更新できる
矛盾をClaimの属性だけでなくConflict objectとして保存できる
Update Logで状態変更を追跡できる
```
