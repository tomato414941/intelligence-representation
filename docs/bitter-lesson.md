# Bitter Lesson Correction

## 役割

この文書は、実装方針書ではなく原則文書である。
現在のデータ形式、モデル構成、評価コマンド、実験状態は扱わない。

ここで確認するのは、プロジェクトが手設計の意味表現システムへ戻らないための判断基準である。

## 問題

このプロジェクトは、放っておくと人間が考えた意味構造を先に固定しやすい。

```text
人間が「知能に必要そうな構造」を先に固定する
  ↓
Entity, Relation, Belief, Goal, Conflict, StateUpdate を大量に作る
  ↓
最初はきれいに見える
  ↓
スケールしない、学習しにくい、硬い、保守できない
```

この方向は避ける。

特に、意味DB、手設計 ontology、retrieval-first memory system を
プロジェクトの中心に戻さない。
それらは将来の補助装置や可視化にはなりうるが、中心仮説そのものではない。

## Correction

中心に置くのは、人間が固定した意味構造ではなく、
データ、計算、学習、評価圧である。

```text
source data
learnable representation
predictive computation
evaluation pressure
```

人間が先に設計すべきなのは、内部の意味構造そのものではない。
モデルが学習できる入出力面、評価対象、失敗が見える検証条件である。

`world model` はこの上位概念の全体ではない。
観測、行動、環境遷移を予測する能力として、
Predictive Representation System の中の評価面に置く。

## What Not To Center

次をプロジェクトの中心にしない。

```text
fixed ontology
handcrafted semantic database
manual belief / conflict taxonomy
hard-coded reasoning rules
elaborate symbolic state machine
retrieval-first memory loop
```

これらは、必要な実験圧が出る前に作ると、Bitter Lesson に反する方向へ戻る。

過去の言い方:

```text
明示的な意味状態メモリを作る
```

避けたい中間案:

```text
検索、要約、外部メモリ、手設計状態を中心にして、
学習モデルをその周辺部品として使う。
```

現在の言い方:

```text
学習可能な表現と予測計算を中心にし、
評価で有用性が出た構造だけを残す。
```

## State Is Not The Source Of Truth

人間可読な `State`、`Belief`、`Claim`、`Conflict` は、
必要なら説明、監査、デバッグ用のビューとして扱う。
それらを最初から真理DBとして固定しない。

```text
source material:
  primary evidence

derived state:
  temporary / revisable view

learned predictive structure:
  what the model must acquire under training and evaluation pressure
```

元データはできるだけ出所に近い形で残す。
そこから作った意味状態は、常に間違いうる派生物として扱う。

## Thin Structure Is Allowed

Bitter Lesson は「構造を一切入れない」という意味ではない。
モデルが学習し、評価が失敗を検出するための薄い境界は入れてよい。

```text
data boundary
prediction boundary
candidate set
train / evaluation split
metric
```

これは ontology ではない。
慎重にすべきなのは、次のような高レベル意味構造を実験前に固定することである。

```text
Entity
Relation
Belief
Goal
Conflict
CausalSchema
PhysicalObjectModel
SocialModel
```

必要なら、それらはモデルが学習、生成、再構成する対象、
または評価後に一時的に取り出す view として扱う。

## Memory And Retrieval

保存、検索、要約、外部メモリは将来の補助装置としては有用になりうる。
ただし、それらを現在の中心にしない。

補助装置を足す条件は、評価で必要性が出たときである。
たとえば長い依存関係、長期記憶、検索対象、読み書き対象が
実験上のボトルネックとして確認された場合に限る。

## Summary

この文書の修正方針は次である。

```text
元データを早く意味DBに変換しない
内部意味構造を人間が先に固定しない
学習可能な表現と予測計算を中心に置く
評価で必要性が出た構造だけを残す
検索や外部メモリは補助装置として扱う
```

このプロジェクトは、意味DBを作るプロジェクトではない。
また、retrieval-first memory system を作るプロジェクトでもない。
