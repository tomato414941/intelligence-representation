# Semantic Memory Architecture

## 目的

Semantic Memoryは、生の観測、抽出された主張、採用中の信念、矛盾、更新履歴を分けて扱う。

目的は、入力を一つの正しい状態へ即座に統合することではない。
元観測を保持しつつ、解釈と信念を更新可能にすることである。

ここでの入力は、人間の自然言語に限らない。
自然言語はObservationの一種であり、画像、音声、センサー値、行動ログ、ツール実行結果、環境状態もObservationになりうる。

ただし、このアーキテクチャは固定された知識表現DBを作ることを目的にしない。
Raw Observationを正本とし、Claim、Belief、Conflict、StateTransitionなどは、検索・推論・説明のために作られる派生キャッシュとして扱う。

## レイヤー

```text
Observation:
  生入力。immutable。source of truth。

Claim:
  Observationから派生した主張。自然言語入力と相性がよい。

Event / StateTransition:
  Observationから派生した出来事や状態変化。非言語観測や世界モデルと相性がよい。

Belief:
  Claim、Event、StateTransitionなどを統合した現在の採用理解。mutable cache。

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
modality
```

Observationは原則として変更しない。
抽出や解釈が間違っていても、元観測が残っていれば再解釈できる。

```text
modality:
  text / image / audio / video / sensor / action_result / tool_output / environment_state
```

## Claim

Claimは、Observationから派生した主張である。
特に自然言語や文書から抽出しやすい意味単位である。

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

ただし、すべてのObservationがClaimに変換されるわけではない。
非言語観測では、Event、State、StateTransition、Affordance、Risk、CausalHypothesisとして表す方が自然な場合がある。

## Event / StateTransition

Eventは、時間を持つ出来事である。
StateTransitionは、状態の変化である。

```text
Event:
  id
  observation_id
  type
  participants
  time
  context

StateTransition:
  id
  observation_id
  before
  after
  cause
  confidence
```

世界モデルや行動ログを扱う場合、ClaimよりもEventやStateTransitionの方が中心になることがある。

## Belief

Beliefは、複数のClaim、Event、StateTransition、Hypothesisを統合した現在の採用理解である。

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
supporting_observations
supporting_events
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
Claim / Event / StateTransitionを作成
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
Beliefは更新可能なキャッシュとして扱う
Conflictは削除ではなく管理対象にする
UpdateLogで変更の理由を追えるようにする
固定スキーマに賭けすぎない
必要な抽象はモデルがその場で構成できるようにする
```
