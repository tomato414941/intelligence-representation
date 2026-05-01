# AIにとって自然な意味表現

## 位置づけ

この文書は、プロジェクトの中心概念を短くまとめる入口である。
実装状態、データ形式、CLI、評価コマンドの正本ではない。

詳細は次の文書に置く。

- [Predictive Representation System](predictive-representation-system.md)
- [Model Input Boundaries](model-input-boundaries.md)
- [Learning and Execution](learning-and-execution.md)
- [Representation, Signal, and Symbol](signal-and-symbol.md)
- [World Model Centering](world-model.md)
- [Bitter Lesson Correction](bitter-lesson.md)
- [Evaluation](evaluation.md)

旧 Semantic State Memory 寄りの背景メモは、現在のコンセプトから切り離して
[Semantic State Memory Background](../legacy/docs/semantic-state-memory-background.md) に移した。

## 中心仮説

AIにとって自然な意味表現は、人間が先に固定した意味DBではなく、
受け取る・生成する・予測する bounded signal の流れの中で形成される。

このプロジェクトの中心は次である。

```text
A predictive representation system for language, perception, action, memory, and belief.
```

ここでの `world model` は全体名ではない。
観測、行動、環境遷移を予測する能力として、
Predictive Representation System の中核的な評価面に位置づける。

## 意味表現の見方

自然言語は重要な入力源だが、唯一の入力源ではない。
画像、音声、センサー、行動結果、tool output、環境状態も、
学習可能な入力と出力の列として扱える。

重要なのは、最初から人間が意味カテゴリを固定することではない。
元データを入力層が扱いやすい形で保持し、
そこからモデル入力に使える embedding sequence へ変換することである。

```text
raw examples
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
```

この形なら、テキスト、画像、音声、行動、選択分類、自由記述応答を、
同じ中間の予測計算へ接続しやすい。
ただし、入力層、出力層、学習目的まで無理に同一化する必要はない。

## 設計原則

優先するもの:

```text
raw examples close to their source
simple input-layer boundaries
input embedding sequences as shared model input
next-token or future-token training as smoke objectives
held-out continuation / ranking / task evaluation
loss reduction and modeling evidence kept separate
```

避けるもの:

```text
handcrafted ontology
fixed Semantic State DB
manual belief / conflict taxonomy
generic envelope schema as the common layer
large schema design before repeated experimental pressure
loss reduction alone as a world-model claim
```

## 観測を起点にする

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

## 構造をどこに置くか

すべてを系列として扱うことは、構造を捨てることではない。
ただし、意味構造そのものを手設計の固定 schema として先に作り込まない。

先に固定してよいのは、学習と評価に必要な境界である。
たとえば、どの raw example をどの tokenizer / encoder に渡すか、
どの範囲を予測対象にするか、どの評価で読むか、といった境界である。

意味のある内部表現は、データ分布、予測対象、評価圧によって
学習されるべきものとして扱う。

## Non-Claims

このリポジトリは、まだ Predictive Representation System を実現していない。
実装状態は README と [Current Evidence](current-evidence.md) を参照する。

まだ示していないもの:

```text
robust action-conditioned future prediction
learned latent predictive state
vision / audio tokenizer integration at scale
belief update
memory read/write learning
large-scale tool-use outcome prediction
planning or control
```

したがって、実験結果は常に限定的に読む。
平均 next-token loss が下がっても、それだけでは world model や汎用的な意味表現が学習されたとは言わない。
