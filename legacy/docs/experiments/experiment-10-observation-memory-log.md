# Experiment 10: Observation Memory Log

## 目的

Observation MemoryにUpdate Logを追加する。

Bitter Lesson寄りの薄い構成を保ちつつ、検索、文脈構成、生成物保存の由来を追えるようにする。

## UpdateLog

```text
id
type
query
input_observations
output_observation
timestamp
```

## 記録する操作

```text
retrieve:
  query と result ids を記録する

build_context:
  input ids を記録する

generated_summary:
  query, input ids, output id を記録する
```

## 例

```text
query:
  Transformer 関係

results:
  obs_1
  obs_2

summary:
  obs_4
```

ログ:

```text
type: generated_summary
query: Transformer 関係
input_observations: [obs_1, obs_2]
output_observation: obs_4
```

## 成功条件

```text
retrieveの結果をログに残せる
build_contextの入力をログに残せる
summaryがどのqueryとobservationsから作られたか追跡できる
```
