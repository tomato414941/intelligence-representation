# AIにとって自然な意味表現

## 位置づけ

この文書は、現在のプロジェクトコンセプトを短くまとめる入口である。
実装方針の正本は次の文書に置く。

- [Predictive Token Machine](predictive-token-machine.md)
- [Signal and Symbol](signal-and-symbol.md)
- [World Model Centering](world-model.md)
- [Bitter Lesson Correction](bitter-lesson.md)
- [Evaluation](evaluation.md)

旧 Semantic State Memory 寄りの背景メモは、現在のコンセプトから切り離して
[Semantic State Memory Background](legacy/semantic-state-memory-background.md) に移した。

## Current Concept

AIにとって自然な意味表現は、人間が先に固定した意味DBではなく、観測・行動・結果・記憶・信念・誤差・tool useを含む signal stream として扱う。

このプロジェクトの中心は次である。

```text
A predictive token machine for language, perception, action, memory, and belief.
```

ここでの `world model` は全体名ではない。
観測、行動、環境遷移を予測する能力として、Predictive Token Machine の中核的な評価面に位置づける。

## Design Direction

優先するもの:

```text
Observation / Action / Consequence を含む signal streams
薄い channel / payload 境界
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

## Signal

このプロジェクトでの `Signal` は、Predictive Token Machine に流す薄い入力・出力単位である。
人間に例えるなら、感覚器官や身体や道具から入ってくる時系列入力に近い。

```text
Signal:
  channel と payload だけを持つ
  text, observation, action, consequence, tool result, memory, belief, error などの単位
```

`Signal` は手設計された意味オブジェクトではない。
モデルが条件づけたり予測したりするための、最小限の envelope として扱う。

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

現在の `channel` 値はレビュー対象であり、確定した分類体系ではない。

| channel | 暫定的な意味 | レビュー観点 |
| --- | --- | --- |
| `text` | 自然言語やコードなど、主にテキストとして扱う信号 | `observation` の text modality と分けるべきか |
| `observation` | 外界、環境、ユーザー、tool 出力などから得た観測 | 広すぎるなら後で細分化する |
| `action` | agent / system が外界や tool に対して行う操作 | tool call と統合すべきか |
| `consequence` | action や状態変化の後に生じた結果・次状態 | observation と分ける粒度 |
| `prediction` | モデルやシステムが明示的に出した予測 | 通常生成と分けるべきか |
| `prediction_error` | 予測と観測・結果の差分、失敗、修正信号 | reward / error と統合すべきか |
| `state` | 環境状態、内部状態、集約状態などの状態表現 | observation / belief との境界 |
| `belief` | 不確実性を含む仮説、推定、信念状態 | state と分ける必要性 |
| `memory` | 記憶として保存・参照される情報単位 | retrieval record に寄りすぎていないか |
| `reward` | 成功、失敗、評価、強化信号 | prediction_error と分ける必要性 |
| `tool_call` | tool や外部関数への呼び出し | action の一種として扱うべきか |
| `tool_result` | tool call の返り値や実行結果 | observation の一種として扱うべきか |

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
