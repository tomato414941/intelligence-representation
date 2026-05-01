# Predictive Representation System

## 中心文

```text
A predictive representation system for language, perception, action, memory, and belief.
```

このプロジェクトの上位概念は、`world model` より広い。

目指しているものは、単に世界状態を予測するモデルではない。自然言語、観測、
行動、映像、音声、状態、信念、記憶、報酬、誤差、tool use を、それぞれの
raw example や interaction record からモデル入力に変換し、Transformer /
attention による予測計算を通じて、汎用的な知的計算を実現できるかを調べる。

この上位仮説を、この文書では `Predictive Representation System` と呼ぶ。

```text
Predictive Representation System:
  multimodal input embedding sequences and hidden states を使って、
  予測・生成・変換・行動選択・観測統合・信念更新・記憶操作を行う
  学習可能な表現システム
```

ここでの system は、CPU やチューリングマシンのような明示的な命令実行装置ではない。
より正確には、入力表現、共有モデル、出力層、目的関数、評価を含む学習可能な予測基盤である。

## 階層

```text
通常の計算機:
  symbol / bit に対する明示的な命令実行

自然言語モデル:
  text token sequence に対する next-token predictor

world model:
  observation / action / consequence history に対する future predictor

Predictive Representation System:
  language, vision, audio, action, state, belief, memory, reward, error,
  tool use を含む表現に対する汎用予測・生成・変換基盤
```

この整理では、world model は全体ではなく一部である。

```text
world model ⊂ predictive representation system
```

`world model` は、観測・行動・環境遷移を予測する能力に焦点を当てた名前である。
一方、`Predictive Representation System` は、それに加えて自然言語、画像、
音声、tool call、記憶、信念、誤差、報酬、自己状態まで含めて扱う上位概念である。

## 機能の分解

`Predictive Representation System` の中には、複数の機能を同じ予測的な表現学習基盤の側面として置ける。

```text
Language Model:
  自然言語を予測・生成・変換する機能

World Model:
  観測・行動・環境遷移を予測する機能

Tool Model:
  tool call と tool result を予測・選択する機能

Belief Model:
  内部状態・不確実性・仮説を表現として扱う機能

Memory Model:
  記憶の読み書き・検索を表現として扱う機能

Reward / Error Model:
  報酬、失敗、誤差、修正、更新信号を表現として扱う機能
```

これらを別々の symbolic system として手設計するのではなく、raw example や
interaction record から作るモデル入力表現と予測学習の上に載せる。

## 基本演算

この system の基本演算は、命令実行ではなく予測である。

```text
予測する
圧縮する
補完する
変換する
生成する
行動を選ぶ
観測を統合する
信念を更新する
tool を呼ぶ
誤差から適応する
記憶を読む
記憶を書く
```

多くの処理は、入力文脈から未知の token、label、state、result、reward、
preference、または次の interaction を推定する問題として表現できる。
ただし、平均 loss が下がることと、上の能力が獲得されたことは別である。

## Training Objective と Evaluation Target

学習形式としては、予測を第一候補にする。

```text
training objective:
  model input context から、未観測または未確定の対象を予測する
```

しかし、評価では全体平均 loss だけを見ない。見るべきなのは、特定の機能が要求される位置である。

```text
action の後の next observation
tool call の後の tool result
partial observation の後の belief-like state
prediction の後の error or correction
memory query の後の memory result
plan の後の outcome
```

`world model` 的能力を見るなら、action-conditioned next-observation prediction を見る。
`tool model` 的能力を見るなら、tool call の結果や適切な tool selection を見る。
`belief model` 的能力を見るなら、partial observation や矛盾する観測の後に内部表現がどう変わるかを見る。

## 表現と構造

すべてを1つの raw schema にすることは、構造を捨てることでも、構造を得ることでもない。
自然言語は token sequence として扱いやすいが、画像、音声、動画、行動履歴は
別の入力層を通って embedding sequence になることが多い。

このプロジェクトで今後中心に置くのは、入力前の汎用 envelope schema ではなく、
raw example / interaction record と、それを input embedding sequence に変換する
modality-specific input layers である。

このプロジェクトで積極的に固定してよいのは、主に学習と評価に実際に使う薄い境界である。

```text
raw example or interaction record
input layer boundary
input embedding sequence boundary
loss mask or objective target
evaluation candidate set
```

これは過剰な ontology ではない。Transformer が予測に必要な構造を学びやすくするための interface である。

避けるべきなのは、人間が最初から次を大量に固定する方向である。

```text
Entity
Relation
Belief
Goal
Ontology
CausalSchema
PhysicalObjectModel
SocialModel
```

人間が設計するべきなのは、内部の意味構造そのものではなく、次である。

```text
input representation
prediction target
objective
evaluation pressure
input / output interface
```

## 入力層は感覚器・行動器に近い

入力層は単なる前処理ではない。世界との接点をモデル入力表現に変換するための
感覚器・行動器・内部状態境界に近い。

```text
language input layer:
  text -> token ids -> token embeddings

vision input layer:
  image / video -> patch or visual embeddings

audio input layer:
  waveform / speech -> audio embeddings

action input layer:
  tool call / motor command / action record -> action representation

state input layer:
  environment state / event logs -> state representation

reward / error input layer:
  reward, failure, correction -> update-related representation
```

新しい入力層を追加することは、新しい感覚器や行動器を追加することに近い。
ただし、追加すれば即座に使えるわけではなく、既存表現との alignment、
行動や観測との因果関係、評価対象との対応を学習または検証する必要がある。

## 現在の位置づけ

現在のリポジトリは、`Predictive Representation System` を実現済みではない。
実装状態の詳細は README と [Current Evidence](current-evidence.md) に置く。

現在の実験基盤は、次の形へ寄せている。

```text
raw examples or interaction records
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
  -> task-specific output layer and evaluation
```

現時点でまだ示していないものは次である。

```text
large-scale multimodal representation learning
vision / audio integration at scale
learned latent predictive state
robust action-conditioned future prediction
belief update
memory read/write learning
large-scale tool-use outcome prediction
reward / error based adaptation
planning or control
```

したがって、今の実験結果は `Predictive Representation System` の証拠ではない。
それに向かうために、入力側の形式を無理に共通化せず、共有できる予測計算へ接続するための最小実験を進めている段階である。

## まとめ

このプロジェクトの上位概念は、`world model` ではなく `Predictive Representation System` である。

```text
Predictive Representation System:
  予測を中心にした、学習可能な表現システム

World Model:
  その中で、観測・行動・環境遷移を予測する能力
```

`world model` という言葉は捨てない。
ただし、それは全体の名前ではなく、Predictive Representation System の中核的な一側面として使う。
