# Semantic Memory Architecture

## 目的

Semantic Memoryは、生の観測、抽出された主張、採用中の信念、矛盾、更新履歴を分けて扱う。

目的は、入力を一つの正しい状態へ即座に統合することではない。
元観測を保持しつつ、解釈と信念を更新可能にすることである。

## レイヤー

```text
Observation:
  生入力。immutable。

Claim:
  Observationから派生した主張。履歴として保持する。

Belief:
  Claimを統合した現在の採用理解。mutable。

Conflict:
  ClaimやBelief間の競合。削除せず管理する。

UpdateLog:
  状態変更の履歴。
```

## Observation

Observationは、外部から入った生情報である。

```text
id
payload
source
created_at
```

Observationは原則として変更しない。
抽出や解釈が間違っていても、元観測が残っていれば再解釈できる。

## Claim

Claimは、Observationから派生した主張である。

```text
id
observation_id
subject
predicate
object
time
context
owner_of_belief
confidence
```

Claimは観測に紐づく解釈であり、Beliefそのものではない。
同じObservationから複数のClaimが出ることもある。

## Belief

Beliefは、複数のClaimを統合した現在の採用理解である。

```text
id
canonical_subject
canonical_predicate
canonical_object
scope
status
confidence
supporting_claims
counter_claims
```

Beliefは更新可能である。
同じ意味のClaimが増えればsupporting_claimsに追加する。
競合するClaimが出ればcounter_claimsやConflictを更新する。

## Conflict

Conflictは、矛盾や競合を第一級オブジェクトとして保存する。

```text
id
left
right
type
status
possible_resolutions
```

矛盾は即削除や即上書きではなく、未解決状態として保持する。

## UpdateLog

UpdateLogは、状態変更の履歴である。

```text
id
type
observation_id
claim_id
target_id
result_id
```

更新型は少なくとも次を持つ。

```text
add_claim
merge_belief
create_conflict
retire_belief
```

## 基本フロー

```text
入力
  ↓
Observationとして保存
  ↓
Claimを作成
  ↓
関連Beliefを検索
  ↓
同等ならBeliefへmerge
  ↓
矛盾ならConflictを作成
  ↓
UpdateLogを残す
```

## 原則

```text
Observationは消さない
Claimは観測由来の解釈として残す
Beliefは更新可能にする
Conflictは削除ではなく管理対象にする
UpdateLogで変更の理由を追えるようにする
```
