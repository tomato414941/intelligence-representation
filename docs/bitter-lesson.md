# Bitter Lesson Correction

## 問題

このプロジェクトは、放っておくと手設計の意味表現システムに寄りやすい。

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

特に、`Semantic State Memory` や `Observation Store + Retrieval + LLM` を
プロジェクトの中心に戻さない。
それらは将来の補助装置になりうるが、現在の中心ではない。

## Current Correction

現在の中心文は次である。

```text
A predictive token machine for language, perception, action, memory, and belief.
```

このプロジェクトの核は、手設計の意味DBでも、retrieval-first memory systemでもない。

```text
raw examples
  ↓
modality-specific tokenization / encoding
  ↓
TokenSequence
  ↓
small decoder-only Transformer predictor
  ↓
next-token training as a smoke objective
  ↓
target-position future prediction evaluation
```

人間が設計するべきなのは、内部意味構造そのものではなく、モデルが学習できる入出力面と評価圧である。

```text
tokenization
serialization
prediction target
negative construction
train/eval split
ranking metric
evaluation pressure
encoder / decoder interface
```

`world model` はこの上位概念の全体ではない。
観測・行動・環境遷移を予測する能力として、Predictive Token Machine の中の評価面に置く。

## What Not To Center

次をプロジェクトの中心にしない。

```text
fixed ontology
handcrafted Semantic State DB
manual belief / conflict taxonomy
hard-coded reasoning rules
elaborate symbolic state machine
retrieval-first LLM memory loop
```

これらは、必要な実験圧が出る前に作ると、Bitter Lesson に反する方向へ戻る。

過去の言い方:

```text
明示的な Semantic State Memory を作る
```

中間的な言い方:

```text
Observation Memory と Retrieval を中心にし、
LLM が必要な文脈から抽象状態を構成する。
```

現在の言い方:

```text
typed streams と learned predictor を中心にし、
future prediction evaluation で有用な構造だけを残す。
```

## State Is Not The Source Of Truth

人間可読な `State`、`Belief`、`Claim`、`Conflict` は、必要なら説明・監査・デバッグ用のビューとして扱う。
それらを最初から真理DBとして固定しない。

```text
Raw / typed observations:
  source material

Derived state:
  temporary / revisable view

Learned predictive structure:
  what the model must acquire under training and evaluation pressure
```

元の観測や signal stream は残す。
そこから作った意味状態は、常に間違いうる派生物として扱う。

## Thin Structure Is Allowed

Bitter Lesson は「構造を一切入れない」という意味ではない。
モデルが予測対象を見つけるための薄い stream interface は入れてよい。

```text
channel
payload
boundary
target channel
```

これは ontology ではない。
Transformer に世界を見せる serialization interface である。

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

必要なら、それらはモデルが学習・生成・再構成する対象、または評価後に一時的に取り出す view として扱う。

## Current Loop

現在の実験ループは retrieval loop ではない。

```text
Generate or collect raw examples
  ↓
Tokenize or encode into TokenSequence
  ↓
Train a small decoder-only predictor
  ↓
Evaluate target-position future prediction
  ↓
Inspect prediction errors and ranking failures
  ↓
Adjust data, rendering, context length, model size, or evaluation
```

平均 next-token loss は smoke signal にすぎない。
world-modeling 的な主張には、action-conditioned next-observation や consequence ranking の改善が必要である。

## Retrieval And Memory

Observation store、retrieval、context builder、summary cache は将来の補助装置としては有用になりうる。
ただし、それらを現在の中心にしない。

位置づけは次である。

```text
primary:
  typed stream prediction and target-position evaluation

support:
  observation storage
  retrieval
  context building
  summaries as cache
  external memory interfaces
```

補助装置を足す条件は、評価で必要性が出たときである。
たとえば context length、data scale、long-horizon dependency、memory read/write target が実験上のボトルネックとして確認された場合に限る。

## Summary

この文書の修正方針は次である。

```text
保存するのは token を作りやすい raw examples
固定するのは TokenSequence と必要な tokenizer / encoder interface
中心に置くのは learned predictor
評価するのは target-position future prediction
抽象化はモデルと評価圧に任せる
手設計 ontology と retrieval-first memory system には戻らない
```

このプロジェクトは、意味DBを作るプロジェクトではない。
また、retrieval + LLM memory system を作るプロジェクトでもない。

現在の中心は、typed token streams 上で予測学習を行い、未来予測が改善するかを測る Predictive Token Machine scaffold である。
