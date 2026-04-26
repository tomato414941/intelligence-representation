# Experiment 09: Observation Memory

## 目的

Bitter Lesson寄りの最小コアを作る。

Stateを固定DBにせず、Observationと検索・文脈構成を中心にする。

```text
Observation Store
  ↓
Retriever
  ↓
Context Builder
  ↓
Transformer / LLM
  ↓
Generated Cache
  ↓
Store again
```

この実験ではLLMはまだ使わない。
検索とWorking Context構築、生成物の再保存だけを確認する。

## Observation

最小Observationは次を持つ。

```text
id
content
modality
source
timestamp
type
links
tags
```

`Claim`、`Belief`、`Conflict`などの強いスキーマは使わない。
必要な抽象は、後段のモデルがその場で作る。

## 入力例

```text
obs_1:
  Transformerは系列モデルというより関係計算器に近い

obs_2:
  Attentionはトークン間の重み付き関係を作る

obs_3:
  State Memoryは真理DBではなくキャッシュとして扱う
```

## 検索例

```text
query:
  Transformer 関係

result:
  obs_1
  obs_2
```

## Working Context

```text
[obs_1] Transformerは系列モデルというより関係計算器に近い
[obs_2] Attentionはトークン間の重み付き関係を作る
```

## Generated Cache

生成された要約や判断もObservationとして保存する。

```text
type: summary
source: generated
links: [obs_1, obs_2]
content:
  Transformerはトークン間関係を計算する機構として見られる。
```

## 成功条件

```text
Observationを保存できる
文字列・タグで検索できる
検索結果からWorking Contextを作れる
生成されたsummaryをObservationとして再保存できる
summaryから元Observationへのlinksを保持できる
```
