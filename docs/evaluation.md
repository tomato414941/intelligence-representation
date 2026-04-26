# Evaluation

## 現在地

現在の主線は、Semantic State Memoryの美しさではなく、世界モデル的な予測評価である。

評価の中心は次である。

```text
観測履歴やretrievalが、次状態予測を改善するか
必要な文脈だけを使えているか
曖昧性・競合・時間差・信頼度を、予測結果として扱えるか
```

Experiment 17-22を現在の主線として扱う。
それ以前のsemantic/state系実験は、historical concept sketchesである。

## 目的

Semantic State Memoryが「良い」と言える条件を明確にする。

意味状態を保持し、関係を明示し、矛盾を管理しても、それが何に対して有効なのかを決めなければ設計判断ができない。

このプロジェクトの初期評価では、まず次の三つを見る。

```text
検索と文脈構成
根拠追跡性
キャッシュの有用性
予測への有用性
状態更新の正しさ
矛盾管理
```

## 評価軸

### 検索と文脈構成

Observation Storeから、現在の問いに必要なObservationを取り出せるかを見る。

例:

```text
入力:
  obs_1: Transformerは系列モデルというより関係計算器に近い
  obs_2: Attentionはトークン間の重み付き関係を作る
  obs_3: State Memoryは真理DBではなくキャッシュとして扱う

query:
  Transformer 関係

期待:
  retrieve: obs_1, obs_2
  context includes obs_1, obs_2
```

この評価は、固定スキーマではなくObservation Memoryを中心にするための基本である。

### キャッシュの有用性

生成されたsummaryやdecisionが、後続の検索や文脈構成で役に立つかを見る。

例:

```text
summary:
  Transformerはトークン間関係を計算する機構として見られる。

links:
  obs_1
  obs_2

期待:
  summaryがObservationとして保存される
  summaryから元Observationへたどれる
  後続queryでsummaryを再利用できる
```

Generated Cacheは真理ではなく、再利用可能な派生物である。

### スキーマ最小性

固定スキーマに依存しすぎていないかを見る。

例:

```text
良い:
  observation, source, timestamp, tags, links, update_log で処理できる

慎重:
  fixed ontology, handcrafted belief rules, hard-coded conflict taxonomy がないと動かない
```

薄いObservation Memoryでできることは、厚いState Schemaに入れる前にそちらで試す。

### 予測への有用性

世界モデルとして見るなら、表現の良さは未来予測や行動結果予測に役立つかで評価する。

例:

```text
入力:
  observation_t
  action_t

期待:
  next_observation を予測できる
  state_transition を予測できる
  risk / reward / uncertainty を推定できる
```

人間可読な構造がきれいかどうかではなく、予測誤差が下がるか、行動選択が良くなるかを見る。

### 状態更新の正しさ

入力後に、期待されるClaim、Belief、Conflict、UpdateLogが作られるかを見る。

例:

```text
入力:
  田中が本を持っている
  田中が佐藤に本を渡した
  佐藤が本を図書館に置いた

期待:
  has(田中, 本): inactive
  has(佐藤, 本): inactive
  located_at(本, 図書館): active
```

### 根拠追跡性

BeliefやClaimから、元になったObservationへたどれるかを見る。

例:

```text
Belief:
  has(田中, 本)

Evidence:
  obs_1
  obs_2
```

根拠が追えないBeliefは、再解釈、検証、rollbackが難しい。

### 矛盾管理

矛盾した入力を削除や上書きで処理せず、Conflictとして保持できるかを見る。

例:

```text
入力:
  田中が本を持っている
  佐藤が本を持っている

期待:
  belief_1: active
  belief_2: uncertain
  conflict_1: unresolved
```

## 現在のテスト対応

### Current Main Line

```text
tests/test_observation_assisted_prediction.py:
  memoryなし / recent memory / retrieved memory を比較し、
  retrieved contextが予測を改善するかを見る

tests/test_multihop_observation_prediction.py:
  直接検索だけでは不足するmulti-hop予測を扱う

tests/test_ambiguous_multihop_prediction.py:
  一意に解けないmulti-hopをambiguous候補集合として扱う

tests/test_temporal_multihop_prediction.py:
  時間差のある観測で、最新観測を現在状態として使い、
  古い観測をsupersededとして残す

tests/test_temporal_conflict_prediction.py:
  同時刻の競合をrecencyで解かず、conflict候補として保持する

tests/test_reliability_weighted_prediction.py:
  source reliabilityで同時刻競合を解ける場合と解けない場合を分ける
```

この系統で見る指標は次である。

```text
prediction accuracy
context size
retrieved observation ids
ambiguous / conflict rate
provenance observation ids
counterevidence observation ids
superseded observation ids
```

### Historical Concept Sketches

```text
tests/test_semantic_state.py:
  構造化イベントから現在状態を更新できるか

tests/test_contextual_claims.py:
  time / context / owner_of_belief によって矛盾候補を区別できるか

tests/test_contextual_state_update.py:
  文脈付きイベントで状態更新と矛盾候補保持を同時に扱えるか

tests/test_observation_belief_conflict.py:
  Observation, Belief, Conflict, UpdateLogを分けられるか

tests/test_semantic_memory.py:
  Observation, Claim, Belief, Conflict, UpdateLogを分離し、ClaimをBeliefへ統合できるか

tests/test_observation_stream.py:
  Observationのmodalityに応じてClaim / Event / StateTransitionを作れるか

tests/test_world_state_update.py:
  StateTransitionをWorldStateへ反映できるか

tests/test_state_hub.py:
  Observationを起点にBeliefState / WorldState / ConflictState / Provenanceを更新できるか

tests/test_observation_memory.py:
  薄いObservation Storeで検索とWorking Context構築ができるか

tests/test_observation_memory_log.py:
  retrieve / build_context / generated_summary の履歴をUpdateLogとして追跡できるか
```

## 最小評価ケース

### Observation Retrieval

```text
入力:
  obs_1: Transformerは系列モデルというより関係計算器に近い
  obs_2: Attentionはトークン間の重み付き関係を作る
  obs_3: State Memoryは真理DBではなくキャッシュとして扱う

query:
  Transformer 関係

期待:
  retrieve: obs_1, obs_2
  context:
    [obs_1] ...
    [obs_2] ...
```

### Generated Cache

```text
入力:
  obs_1
  obs_2

生成:
  summary: Transformerはトークン間関係を計算する機構として見られる。

期待:
  summary is Observation
  summary.type: summary
  summary.source: generated
  summary.links: [obs_1, obs_2]
```

### Update Log

```text
操作:
  retrieve
  build_context
  store_generated_summary

期待:
  update_log includes retrieve
  update_log includes build_context
  update_log includes generated_summary
  generated_summary log has query, input_observations, output_observation
```

### 重複統合

```text
入力:
  田中が本を持っている
  田中が本を持っている

期待:
  Observation: 2
  Claim: 2
  Belief: 1
  Belief.supporting_claims: 2
  UpdateLog: add_claim, merge_belief
```

### 矛盾保持

```text
入力:
  田中が本を持っている
  佐藤が本を持っている

期待:
  Observation: 2
  Claim: 2
  Belief: 2
  Conflict: 1
  UpdateLog: add_claim, create_conflict
```

### 信念主体の分離

```text
入力:
  世界状態として、田中が本を持っている
  田中の信念として、佐藤が本を持っている

期待:
  Conflict: 0
```

同じ対象について違う主張でも、信念主体が違うなら矛盾とは限らない。

## 将来の評価軸

```text
長期一貫性:
  長い入力列のあとでも、重要なBeliefが壊れないか。

再解釈可能性:
  抽出や解釈が誤っていたとき、Raw Observationから再構成できるか。

検索効率:
  必要なBelief / Claim / Evidenceを少ないコストで取り出せるか。

圧縮品質:
  多数の観測を、根拠を失わずに少数のBeliefへまとめられるか。

誤書き込み耐性:
  低信頼・誤抽出・敵対的入力が長期状態を汚染しにくいか。

行動・検証への有用性:
  Beliefが検索、ツール実行、実験、外部観測によって検証・更新できるか。

予測性能:
  次状態、次観測、行動結果、不確実性、リスクを予測できるか。

能動的観測:
  不確実性を減らすための検索、質問、実験、ツール実行を選べるか。

スキーマ進化:
  新しい種類の意味単位を追加しても、過去データを壊さず扱えるか。

スキーマ最小性:
  厚い手設計スキーマを追加せず、Observation + Retrieval + Contextで処理できる範囲を広げられるか。
```

## 方針

次の実験からは、実装前に評価ケースを書く。

特に自然言語入力へ進む場合も、評価対象は自然文そのものではなく、最終的に得られるObservation、Claim、Belief、Conflict、UpdateLogが期待通りかで見る。

ただし、自然言語はObservationの一種でしかない。
画像、センサー、行動結果、ツール出力を扱う場合は、Claimだけでなく、Entity、Event、StateTransition、Hypothesis、Evidenceが期待通りに作られるかも評価対象にする。

Bitter Lesson寄りの方針では、まずObservation Memory、Retrieval、Working Context、Generated Cache、Update Logを評価する。
Claim、Belief、Conflictなどの厚い構造は、有用性が確認できた場合にキャッシュとして導入する。

ただし、最終的な評価軸は外部メモリの整然さではなく、世界モデルとしての予測、行動結果予測、不確実性低減、状態更新の改善である。
