# AIにとって自然な意味表現

## 位置づけ

この文書は、現在のプロジェクトコンセプトを短くまとめる入口である。
実装方針の正本は次の文書に置く。

- [Predictive Token Machine](predictive-token-machine.md)
- [Token Sequence Direction](token-sequence-direction.md)
- [Signal and Symbol](signal-and-symbol.md)
- [World Model Centering](world-model.md)
- [Bitter Lesson Correction](bitter-lesson.md)
- [Evaluation](evaluation.md)

旧 Semantic State Memory 寄りの背景メモは、現在のコンセプトから切り離して
[Semantic State Memory Background](legacy/semantic-state-memory-background.md) に移した。

## Current Concept

AIにとって自然な意味表現は、人間が先に固定した意味DBではなく、
受け取る・生成する・予測する bounded signal の流れとして扱う。
記憶、信念、誤差、tool use は重要な対象だが、現時点では固定 channel
として先に作り込まない。

このプロジェクトの中心は次である。

```text
A predictive token machine for language, perception, action, memory, and belief.
```

ここでの `world model` は全体名ではない。
観測、行動、環境遷移を予測する能力として、Predictive Token Machine の中核的な評価面に位置づける。

## Design Direction

優先するもの:

```text
token を作りやすい raw examples
TokenSequence を共通学習入力にする
next-token training as a smoke objective
continuation / future-token evaluation
held-out hard-negative ranking
loss reduction と modeling evidence の分離
```

避けるもの:

```text
handcrafted ontology
fixed Semantic State DB
manual belief / conflict taxonomy
large schema design before repeated experimental pressure
generic Signal JSONL growth path
loss reduction alone as a world-model claim
```

## Signal

このプロジェクトでの `Signal` は、過去の実験で使っている薄い入力・出力単位である。
人間に例えるなら、感覚器官や身体や道具から入ってくる時系列入力に近い。

```text
Signal:
  channel と payload だけを持つ
  channel 値はデータセットと評価が決める実用ラベルであり、コード上の固定分類ではない
```

`Signal` は手設計された意味オブジェクトではない。
モデルが条件づけたり予測したりするための、最小限の envelope として扱う。
ただし、新しい本線では `Signal` を成長させず、raw example から
`TokenSequence` を作る方向へ寄せる。

## Channel

`channel` は、Signal をどの入力経路・tokenizer・評価経路に流すかを表す実用ラベルである。
以前は `role` と `modality` を分けていたが、現時点では過剰なので 1 つにまとめる。

```text
channel = tokenizer / encoder / evaluation routing のための入力経路
payload = tokenizer / encoder に渡される可変長の実データ

現在の byte-tokenizer 系 renderer は、text / code / structured JSON / action を text payload として扱う。
構造化 action などは Signal に入れる前に canonical JSON text にする。
非テキストメディアは `payload_ref` として保存できる。
`payload_ref` は画像などを学習対象・条件入力として扱うための dataset/corpus 層の参照境界である。
ただし現在の byte-tokenizer training path には channel-specific loader / encoder がまだ接続されていないため、`payload_ref` はそこで明示的に拒否する。
```

現時点では core channel list をコードに持たない。
`text`, `image`, `action`, `observation`, `consequence` のような語は、
固定 ontology ではなく、データセット生成・encoder 選択・評価 target 指定のための
ローカルなラベルとして扱う。

特に次の語は、扱いたい対象ではあっても固定 channel として先に作り込まない。

```text
tool_call / tool_result:
  当面は action や observation/consequence の特殊例として扱う

prediction_error / state / belief / memory / reward:
  重要な対象だが、現時点では channel として固定しない
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
channel
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
現在あるのは、小さな signal stream scaffold と評価系である。

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
