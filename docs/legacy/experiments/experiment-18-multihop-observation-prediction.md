# Experiment 18: Multi-Hop Observation Prediction

## 目的

Observationをそのままコピーするだけでは当たらない予測ケースを作る。

Experiment 17では、retrieveされたObservationに正解factが直接含まれていた。
Experiment 18では、複数のObservationをたどらないと正解できないケースを扱う。

## 例

```text
obs_1:
  located_at(鍵, 箱)

obs_2:
  located_at(箱, 棚)

action:
  find(太郎, 鍵, unknown)

expected:
  located_at(鍵, 棚)
```

## 比較条件

```text
no_memory:
  memoryなし

direct_memory:
  action.objectに直接関係するObservationだけ使う

multi_hop_memory:
  retrieved observation の object を次のqueryとして使い、関係を追加でたどる
```

## 評価

```text
accuracy
context_size
retrieved_observation_ids
case別の正誤
```

## 意味

この実験はまだ学習器ではない。

ただし、次の違いを明示できる。

```text
直接検索だけでは足りないケースがある
関係を複数段たどると予測が改善する
contextを増やすことには意味があるが、必要な関係を選ぶ必要がある
```

## 成功条件

```text
no_memory: 失敗
direct_memory: located_at(鍵, 箱) までは出るが expected とは不一致
multi_hop_memory: located_at(鍵, 棚) を予測できる
```
