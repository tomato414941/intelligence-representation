# Bitter Lesson Correction

## 問題

Semantic State Memoryは、放っておくと手設計の知識表現システムに寄りすぎる。

```text
人間が「知能に必要そうな構造」を細かく設計する
  ↓
Entity, Relation, Belief, Goal, Conflict, StateUpdateを大量に作る
  ↓
最初はきれいに見える
  ↓
スケールしない、学習しにくい、硬い、保守できない
```

この方向は避ける。

## 方針

細かい意味構造を最初から固定するのではなく、モデルが学習・生成・再構成できる一般的な枠を用意する。

最小構造は次でよい。

```text
Observation Store
Retriever
Working Context Builder
Transformer / LLM
Update Log
```

つまり、

```text
観測を保存する
必要なものを検索する
現在の文脈を構成する
モデルに推論させる
結果をログとして残す
```

を中心にする。

## Stateは真理DBではない

Stateを真理のデータベースにしない。
Stateは観測から作られるキャッシュである。

```text
Raw Observation:
  source of truth

Derived State:
  temporary / revisable cache
```

元の観測は残す。
そこから作った意味状態は、常に間違いうる派生物として扱う。

## 薄いメタデータ

最初に保存する構造は薄くする。

```text
id
content / payload
timestamp
source
modality
embedding
links
type: observation | summary | decision | question | artifact
```

`Claim`、`Belief`、`Conflict`、`Goal`などは、固定DBスキーマとして先に作り込みすぎない。
必要なときにモデルが抽出・比較・要約できればよい。

## 残してよい構造

```text
observation
retrieval
timestamp
source
embedding
lightweight links
summaries as cache
update log
```

## 慎重にすべき構造

```text
fixed ontology
handcrafted belief system
manual conflict taxonomy
hard-coded reasoning rules
elaborate symbolic state machine
```

## 実行ループ

```text
Observation Store
  ↓
Retriever / Router
  ↓
Context Builder
  ↓
Transformer / LLM
  ↓
Generated Summary / Decision / Next Action
  ↓
Store again
```

このループでは、抽象状態は固定された真理ではなく、その時点のタスクに合わせて構成される作業表現である。

## このプロジェクトでの修正

前の言い方:

```text
明示的な Semantic State Memory を作る
```

修正後:

```text
Observation Memory と Retrieval を中心にし、
意味状態は固定スキーマではなく、
モデルが必要に応じて構成する一時的表現として扱う。
```

つまり、このプロジェクトの核は、細かい意味DBを作ることではない。

```text
LLMが大量の観測履歴から、
必要な文脈を取り出し、
その場で適切な抽象状態を構成できるようにする。
```

## まとめ

強いモデル、薄い外部記憶、学習可能な検索・圧縮・文脈構成を中心にする。

```text
保存するのは観測
固定するのは最小限のメタデータ
抽象化はモデルに任せる
状態はキャッシュとして扱う
評価で有用なものだけ残す
```
