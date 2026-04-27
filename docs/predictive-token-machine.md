# Predictive Token Machine

## 中心文

```text
A predictive token machine for language, perception, action, memory, and belief.
```

このプロジェクトの上位概念は、`world model` より広い。

目指しているものは、単に世界状態を予測するモデルではない。
自然言語、観測、行動、映像、音声、状態、信念、記憶、報酬、誤差、tool useをtyped token streamに落とし、Transformer / attentionによる予測計算を通じて、汎用的な知的計算を実現できるかを調べる。

この上位仮説を、この文書では `Predictive Token Machine` と呼ぶ。

```text
Predictive Token Machine:
  typed multimodal token stream上で、
  予測・生成・変換・行動選択・観測統合・信念更新・記憶操作を行う
  学習可能な汎用token計算機
```

ただし、ここで言う計算機はCPUやチューリングマシンのような、明示的な命令列をdeterministicに実行する装置ではない。
より正確には、token stream上の確率的・予測的・学習可能な計算基盤である。

## 階層

```text
通常の計算機:
  symbol / bitに対する明示的な命令実行

自然言語モデル:
  text token streamに対するnext-token predictor

world model:
  observation / action / consequence streamに対するfuture predictor

Predictive Token Machine:
  text, vision, audio, action, state, belief, memory, reward, error, tool useを含む
  typed multimodal token streamに対する汎用予測・生成・変換計算機
```

この整理では、world modelは全体ではなく一部である。

```text
world model ⊂ predictive token machine
```

`world model` は、観測・行動・環境遷移を予測する能力に焦点を当てた名前である。
一方、`Predictive Token Machine` は、それに加えて自然言語、画像、音声、tool call、記憶、信念、誤差、報酬、自己状態まで含めて扱う上位概念である。

## 機能の分解

`Predictive Token Machine` の中には、複数の機能を同じtoken prediction基盤の側面として置ける。

```text
Language Model:
  自然言語token streamを予測・生成・変換する機能

World Model:
  観測・行動・環境遷移を予測する機能

Tool Model:
  tool callとtool resultを予測・選択する機能

Belief Model:
  内部状態・不確実性・仮説をtokenとして扱う機能

Memory Model:
  記憶の読み書き・検索をtoken streamとして扱う機能

Reward / Error Model:
  報酬、失敗、誤差、修正、更新信号をtokenとして扱う機能
```

これらを別々のsymbolic systemとして手設計するのではなく、共通のtyped token streamと予測学習の上に載せる。

## 基本演算

この機械の基本演算は、命令実行ではなく予測である。

```text
予測する
圧縮する
補完する
変換する
生成する
行動を選ぶ
観測を統合する
信念を更新する
toolを呼ぶ
誤差から適応する
記憶を読む
記憶を書く
```

多くの処理は、token streamにserializeすればnext-token / future-token predictionとして表現できる。
ただし、平均next-token lossが下がることと、上の能力が獲得されたことは別である。

## Training ObjectiveとEvaluation Target

学習形式としては、next-token predictionを第一候補にする。

```text
training objective:
  typed multimodal token stream上のnext-token / future-token prediction
```

しかし、評価では全token平均lossだけを見ない。
見るべきなのは、特定の機能が要求される位置である。

```text
<ACTION> の後の <OBS>
<TOOL_CALL> の後の <TOOL_RESULT>
<OBS> の後の <BELIEF>
<PREDICTION> の後の <ERROR>
<MEMORY_READ> の後の <MEMORY_RESULT>
<PLAN> の後の <OUTCOME>
```

`world model` 的能力を見るなら、action-conditioned next-observation predictionを見る。
`tool model` 的能力を見るなら、tool callの結果や適切なtool selectionを見る。
`belief model` 的能力を見るなら、partial observationや矛盾する観測の後にbeliefがどう変わるかを見る。

## Token Streamと構造

すべてをtoken streamにすることは、構造を捨てることではない。
自然言語も一次元のtoken streamだが、その中には文法、照応、時間、因果、目的、信念、社会関係が含まれる。

このプロジェクトで積極的に入れてよいのは、主に薄いstream構造である。

```text
modality
time
role
boundary
agent
source
prediction target
action / observationの区別
```

これは過剰なontologyではない。
Transformerが予測に必要な構造を学びやすくするためのinterfaceである。

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
tokenization
serialization
prediction target
evaluation pressure
encoder / decoder interface
```

## Tokenizerは感覚器・行動器である

tokenizerは単なる文字列分割器ではない。
世界との接点をtoken化するための感覚器・行動器・内部状態チャネルに近い。

```text
language tokenizer:
  text -> text tokens

vision tokenizer:
  image / video -> visual tokens

audio tokenizer:
  waveform / speech -> audio tokens

action tokenizer:
  tool call / motor command -> action tokens

state tokenizer:
  environment state / event logs -> state tokens

belief tokenizer:
  belief report / latent predictive state -> belief tokens

reward / error tokenizer:
  reward, failure, correction -> update-related tokens
```

新しいtokenizerを追加することは、新しい感覚器や行動器を追加することに近い。
ただし、追加すれば即座に使えるわけではなく、既存token空間との対応、embedding、自然言語概念とのalignment、行動や観測との因果関係を学習する必要がある。

## 現在の実装位置

現在のリポジトリは、`Predictive Token Machine` を実現済みではない。
まだ小さなscaffoldである。

現時点であるものは次である。

```text
byte-level text tokenizer
small decoder-only GPT
mixed text / environment / grid / log-like corpus
TypedEvent envelope and typed-tag stream rendering
typed-event JSONL input path with legacy MixedDocument compatibility
symbolic-to-natural ranking
action-conditioned next-observation ranking
generated train/eval slices
RunPod execution path
```

現時点でまだ示していないものは次である。

```text
large-scale multimodal token learning
vision / audio tokenizer integration
learned latent predictive state
robust action-conditioned future prediction
belief update
memory read/write learning
tool-use outcome prediction
reward / error based adaptation
planning or control
```

したがって、今の実験結果は `Predictive Token Machine` の証拠ではない。
それに向かうための小さな typed token stream scaffold ができてきた、という位置づけで扱う。

## まとめ

このプロジェクトの上位概念は、`world model` ではなく `Predictive Token Machine` である。

```text
Predictive Token Machine:
  予測を基本演算にした、学習可能なtoken計算機

World Model:
  その中で、観測・行動・環境遷移を予測する能力
```

`world model` という言葉は捨てない。
ただし、それは全体の名前ではなく、Predictive Token Machineの中核的な一側面として使う。
