# Experiment 12: Retrieval Evaluation

## 目的

Retrieverを差し替えて比較できる評価ハーネスを作る。

Bitter Lesson寄りの方針では、検索ロジックを手作りケースに合わせて作り込むのではなく、Retrieverを交換可能にし、評価データを増やせる形にする。

## 評価対象

```text
Retriever:
  retrieve(query, observations, limit) -> observations
```

このインターフェースに合わせれば、n-gram、embedding、hybrid、rerankerを同じ評価ケースで比較できる。

## 評価ケース

```text
EvaluationCase:
  name
  observations
  query
  expected_relevant_ids
  k
```

最小ケースはsmoke testとして扱う。
特定の3文に過適合するためではなく、評価の配線が機能することを確認する。

## 指標

```text
precision_at_k:
  取得結果のうち、期待relevantに含まれる割合

recall_at_k:
  期待relevantのうち、取得できた割合
```

## 成功条件

```text
評価ケースを外部データとして増やせる
Retrieverを差し替えられる
n-gram retrieverをAdapter経由で評価できる
precision_at_k / recall_at_k を計算できる
```
