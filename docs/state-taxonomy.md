# State Taxonomy

## 目的

AIが更新すべきStateを分類する。

これまで「Semantic State」と呼んできたものは、自然言語や議論を扱うには有用だが、知能一般や世界モデルを考えるには狭い。
より一般には、AIはObservation Streamから複数種類のStateを更新する。

```text
Observation
  ↓
StateUpdate
  ↓
World State / Belief State / Task State / Semantic State / Memory State / ...
```

重要なのは、AIが直接持てるのは世界そのものではなく、観測から構成した推定状態であることだ。
World Stateも、本当の世界状態ではなくEstimated World Stateである。

## 最小構成

最小構成として、まず次を扱う。

```text
Observation State:
  何を観測したか

Entity / Object State:
  何が存在するか

Relation State:
  何と何が関係しているか

Event / StateTransition State:
  何が起き、何が変わったか

Belief / Hypothesis State:
  何をどの程度信じているか

Conflict State:
  何が矛盾・競合しているか

Task / Goal State:
  何を達成しようとしているか

Memory / Provenance State:
  根拠はどこにあるか
```

この最小構成があれば、自然言語、世界モデル、ツール実行、長期議論にある程度対応できる。

## Observation State

生の観測を保持する状態である。

```text
自然言語発話
画像
動画
音声
センサー値
ツール実行結果
APIレスポンス
行動ログ
ユーザー操作
```

Observationは、後から再解釈できる元データである。
基本的には消さない。

## World / Environment State

外界についての推定状態である。

```text
何が存在するか
どこにあるか
どんな状態か
何が起きたか
何が変化したか
```

物理世界だけでなく、ソフトウェア環境、会話環境、ツール環境にも適用できる。

例:

```text
cup_17 is on table
door_3 is open
battery_level = 42%
API endpoint is unavailable
現在の話題はTransformerの計算効率
```

## Entity / Object State

対象単位の状態である。

```text
人
物体
概念
文書
ツール
モデル
ファイル
タスク
仮説
```

Entity Stateがあると、関係や状態変化を扱いやすくなる。

例:

```text
entity_id: concept_transformer
type: concept
label: Transformer
attributes:
  role: token relation computation architecture
```

## Relation State

対象同士の関係である。

```text
A supports B
A contradicts B
A causes B
A is part of B
A depends on B
A updates B
A refers to B
A is evidence for B
```

Relation Stateは、Semantic Stateの中心でもあり、World Stateの構造でもある。

## Event / StateTransition State

出来事と状態変化である。

```text
何が起きたか
いつ起きたか
その前後で何が変わったか
何が原因らしいか
```

知能を世界状態系列と結びつけるなら、EventとStateTransitionは中心的である。
自然言語だけを扱うとClaimに寄りやすいが、世界モデルではEventとStateTransitionの方が重要になることがある。

## Belief State

AIが何をどの程度信じているかである。

World StateとBelief Stateは分ける必要がある。

```text
実際の世界状態
AIが信じている状態
ユーザーが信じている状態
別エージェントが信じている状態
```

Belief Stateは少なくとも次を持つ。

```text
statement
confidence
evidence
counterevidence
scope
timestamp
status
```

## Hypothesis / Uncertainty State

不確実性と複数仮説を管理する状態である。

AIは無理に一つの答えへ潰さず、未解決の競合状態を保持できるべきである。

```text
hypothesis A: 0.6
hypothesis B: 0.3
unknown: 0.1
```

## Conflict State

矛盾や競合を明示的に管理する状態である。

矛盾した入力が来たときに、古い情報を即座に消すのではなく、まずConflictとして保持する。

```text
事実矛盾
時間差による変化
文脈差
視点差
抽象度の違い
解釈ミス
曖昧な自然言語
```

## Semantic State

概念、主張、意味構造の状態である。

```text
Concept
Claim
Argument
Definition
Distinction
Analogy
Question
Open Problem
```

自然言語や議論を扱うときには重要だが、全体の中心にしすぎると自然言語寄りになる。
より広い設計では、Semantic Stateは複数Stateの一部である。

## Task State

現在のタスクの状態である。

```text
何を達成しようとしているか
どこまで進んだか
次に何をするべきか
制約は何か
成功条件は何か
```

長い作業をするAIには必須である。

## Goal / Preference State

目的、選好、評価基準である。

```text
正確さを重視する
計算効率を重視する
人間可読性を重視する
安全性を重視する
実装容易性を重視する
```

目的が変わると、保持すべきStateも変わる。

## Action / Policy State

AIが何をしたか、次に何をするか、どの行動が可能かである。

```text
可能な行動
選択済みの行動
行動の予測結果
行動後の観測
```

知能を状態を変える能力として見るなら、Action Stateは外せない。

## Affordance State

対象に対して何ができるかである。

```text
この物体は掴める
このドアは開けられる
このAPIは呼び出せる
このファイルは編集できる
この主張は検証できる
```

世界理解を行動に接続するにはAffordance Stateが必要になる。

## Memory / Provenance State

どの情報がどこから来たかである。

```text
source
timestamp
origin
evidence
derived_from
updated_by
```

信頼性、検証、rollbackに必須である。

## User / Social State

人間と協働するAIに必要な状態である。

```text
ユーザーの関心
ユーザーの前提知識
ユーザーの好み
会話の文脈
合意したこと
約束
未回答の依頼
```

## Norm / Safety / Constraint State

規範、制約、安全性、許可である。

```text
してよいこと
してはいけないこと
確認が必要なこと
安全上の制約
法的・倫理的制約
```

人間社会の中で動くAIには必要になる。

## Self / Capability State

AI自身についての状態である。

```text
何ができるか
何ができないか
どのツールが使えるか
どの情報にアクセスできるか
現在の不確実性
計算資源
```

自己状態がないと、AIは自分の能力や制限を適切に扱えない。

## Resource / Compute State

計算資源、時間、メモリ、レイテンシである。

```text
利用可能なコンテキスト長
検索コスト
外部DBのレイテンシ
推論時間
GPUメモリ
トークン予算
```

実装では、FLOPsだけでなく、I/O、検索、DB更新、分散同期が支配的になることがある。

## まとめ

AIが更新すべきStateは、一つのSemantic Stateではなく、複数の状態の束である。

```text
Observation:
  何を見たか

World / Entity / Relation:
  世界をどう構造化しているか

Event / StateTransition:
  何が変わったか

Belief / Hypothesis:
  何をどの程度信じているか

Conflict:
  何が競合しているか

Task / Goal:
  何を達成しようとしているか

Action / Affordance:
  何ができ、何をするか

Memory / Provenance:
  根拠はどこか

User / Social:
  誰とどう関わっているか

Norm / Constraint:
  何が許され、何が制約されるか

Self / Resource:
  AI自身と計算資源の状態
```

このプロジェクトで作ろうとしているものは、LLMを単なる自然言語生成器として使うのではなく、観測から世界・意味・信念・矛盾・目的の状態を更新し続けるシステムである。

短く言えば、Transformerの外側に、更新可能な意味・世界状態を持つAIアーキテクチャを作ることである。
