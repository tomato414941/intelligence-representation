# AIにとって自然な意味表現

## 位置づけ

この文書は、現在のプロジェクトコンセプトを短くまとめる入口である。
実装方針の正本は次の文書に置く。

- [Predictive Token Machine](predictive-token-machine.md)
- [World Model Centering](world-model.md)
- [Bitter Lesson Correction](bitter-lesson.md)
- [Evaluation](evaluation.md)

旧 Semantic State Memory 寄りの背景メモは、現在のコンセプトから切り離して
[Semantic State Memory Background](legacy/semantic-state-memory-background.md) に移した。

## Current Concept

AIにとって自然な意味表現は、人間が先に固定した意味DBではなく、観測・行動・結果・記憶・信念・誤差・tool useを含む typed token stream として扱う。

このプロジェクトの中心は次である。

```text
A predictive token machine for language, perception, action, memory, and belief.
```

ここでの `world model` は全体名ではない。
観測、行動、環境遷移を予測する能力として、Predictive Token Machine の中核的な評価面に位置づける。

## Design Direction

優先するもの:

```text
Observation / Action / Consequence を含む typed streams
薄い role / modality / time / source 境界
next-token training as a smoke objective
target-position-aware future prediction evaluation
held-out hard-negative ranking
loss reduction と modeling evidence の分離
```

避けるもの:

```text
handcrafted ontology
fixed Semantic State DB
manual belief / conflict taxonomy
large schema design before repeated experimental pressure
loss reduction alone as a world-model claim
```

## Observation First

人間の自然言語は重要な入力源だが、唯一の入力源ではない。
より一般には、入力は observation stream である。

```text
Observation Stream
  ├─ human language
  ├─ image / video
  ├─ audio
  ├─ sensor
  ├─ action result
  ├─ tool output
  └─ environment state
```

自然言語は、人間が世界を観測・解釈し、圧縮した強力な stream である。
ただし、world-modeling 的な評価では、行動後の観測や結果を直接予測できるかを見る必要がある。

## Token Stream And Structure

すべてを token stream にすることは、構造を捨てることではない。
このプロジェクトで先に固定してよい構造は、Transformer が予測対象を見つけやすくするための薄い interface に限る。

```text
role
modality
time
boundary
source
episode
prediction target
```

意味構造そのものは、手設計の固定 schema として先に作り込まない。
必要な構造は、データ分布、予測対象、評価圧によって学習されるべきものとして扱う。

## Current Non-Claims

このリポジトリは、まだ Predictive Token Machine を実現していない。
現在あるのは、小さな typed token stream scaffold と評価系である。

まだ示していないもの:

```text
robust action-conditioned future prediction
learned latent predictive state
vision / audio tokenizer integration
belief update
memory read/write learning
large-scale tool-use outcome prediction
planning or control
```

したがって、実験結果は常に限定的に読む。
平均 next-token loss が下がっても、それだけでは world model や汎用的な意味表現が学習されたとは言わない。
