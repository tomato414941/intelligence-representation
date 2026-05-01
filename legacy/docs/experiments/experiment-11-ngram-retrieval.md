# Experiment 11: N-gram Retrieval

## 目的

外部依存なしで、日本語のような空白なしテキストでもObservationを検索できるようにする。

Experiment 09/10の検索は、空白区切りの語とタグに依存していた。
しかし日本語では空白がないため、タグなしでは検索が弱い。

## 方針

```text
char bigram / trigram を作る
query と content の n-gram overlap でスコアリングする
タグ一致は強めに加点する
```

## 例

```text
content:
  Transformerは系列モデルというより関係計算器に近い

query:
  関係計算

期待:
  retrieveできる
```

## 成功条件

```text
空白なし日本語テキストを検索できる
タグ一致を強めに評価できる
検索結果からWorking Contextを作れる
```
