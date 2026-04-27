# World Model Centering

## 中心文

```text
World modeling as prediction over typed multimodal token streams.
```

このプロジェクトは、AIの内部表現を人間が先に設計することを中心にしない。
また、自然言語モデルとworld modelを別物として対立させることもしない。

中心仮説は次である。

```text
自然言語・観測・行動・映像・音声・状態・信念・記憶・報酬・誤差・tool useを
typed token streamとして表現し、その未来を予測することで、
world model的な表現がどこまで獲得されるかを調べる。
```

自然言語モデルは、world modelの敵でも代替物でもない。
自然言語という特殊なtoken stream上で学習された、world-model-like predictorの一形態として扱う。

より一般には、次の包含関係で見る。

```text
Autoregressive token predictor
  ├─ Natural language model
  │    └─ human text streamのnext-token predictor
  │
  └─ World model-like trajectory model
       └─ observation/action/consequence streamのfuture-token predictor
```

この整理では、world modelを特殊な別アーキテクチャとして最初から作るのではなく、自然言語モデルを含むautoregressive predictorを、世界・行動・観測・信念まで含むtyped stream predictorへ拡張する。

## 学習目的と評価目的

Action-conditioned world predictionは、token streamにserializeすればnext-token predictionとして表現できる。

例えば次の履歴がある。

```text
<OBS> room contains box_a and box_b </OBS>
<OBS> key is inside box_a </OBS>
<OBS> coin is inside box_b </OBS>
<ACTION> open box_b </ACTION>
<OBS> see coin </OBS>
```

この場合、`<OBS> see coin </OBS>`を予測することは形式的にはnext-token predictionである。
したがって、world model的なpredictionをnext-token trainingで実装する方針は筋がよい。

ただし、次の2つは違う。

```text
next-token predictionに還元できる
```

```text
平均next-token lossが下がればworld modelができたと言える
```

モデルは、テンプレートの暗記、頻出語の予測、文体の模倣、固定フォーマットの補完だけでもlossを下げられる。
そのため、このプロジェクトでは次の分離を守る。

```text
training objective:
  next-token predictionを第一候補にする

evaluation target:
  action-conditioned next-observation / future-token predictionを見る
```

特に重要なのは、`<ACTION>`の後の`<OBS>`、`<TOOL_CALL>`の後の`<TOOL_RESULT>`、`<PREDICTION>`の後の`<ERROR>`のような、世界・行動・結果の関係が問われる位置である。

## Token Streamと構造

すべてをtoken streamにすることは、構造を捨てることではない。
自然言語も一次元のtoken streamだが、その中には文法、照応、時間、因果、目的、信念、社会関係などの構造が含まれる。

このプロジェクトで積極的に入れてよい構造は、主に薄いstream構造である。

```text
Level 1:
  stream formatの構造
  modality, role, boundary, time, position, agent, prediction target flag

Level 2:
  data distribution上の構造
  open boxの後には箱の中身が見えることがある

Level 3:
  model内部のlearned structure
  embedding geometry, attention pattern, latent state, internal circuit

Level 4:
  人間が手設計したsymbolic structure
  Entity, Relation, Belief, Goal, Ontology
```

Level 1の型・境界・時間・modality情報は、Transformerが予測に必要な関係を学びやすくするためのinterfaceである。
一方、Level 4の意味構造を大量に手設計する方向には寄せない。

人間が設計するべきなのは、内部の意味構造そのものではなく、次である。

```text
tokenization
serialization
prediction target
evaluation pressure
encoder / decoder interface
```

これはThe Bitter Lessonと相性がよい。
人間がontologyを細かく固定するのではなく、スケールする予測学習に構造を獲得させる。

## Tokenizerの位置づけ

新しいtokenizerを追加することは、新しい感覚器・行動器・内部状態チャネルを追加することに近い。

```text
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
  feedback / loss / correction -> update-related tokens
```

ただし、新しいtokenizerを足せば即座に使えるわけではない。
既存token空間との対応、新modalityのembedding、自然言語概念とのalignment、行動や観測との因果関係を学習する必要がある。

現在の実装は、byte-level text、symbolic environment episode、grid observation、log-like textを扱う足場である。
まだvision/audio/reward/error/tool-useを統合したmultimodal world modelではない。

## 目的

プロジェクトの中心を、外部メモリやSemantic State Memoryではなく、世界モデルに置き直す。

外部メモリは重要だが主役ではない。
主役は、観測と行動から内部状態を作り、未来や行動結果を予測し、誤差によって状態を更新するモデルである。

## 中心ループ

```text
Observation
  ↓
Latent State
  ↓
Prediction
  ↓
Action / Query / Tool Use
  ↓
New Observation
  ↓
Prediction Error
  ↓
State Update
```

世界モデルの本質は、状態を保存することではなく、未来と行動結果を予測し、誤差によって表現を更新することである。

## 形式

```text
z_t = encode(observation_history, action_history)

z_{t+1} = transition(z_t, action_t)

prediction = decode_or_predict(z_{t+1})
```

`z_t` は世界の真の状態ではない。
AIが観測から作る潜在的な推定状態である。

## Predictive State

`Semantic State` は自然言語寄りである。
世界モデル寄りには、中心は `Predictive State` である。

```text
Semantic State:
  何を意味しているか
  どの主張があるか
  何が矛盾しているか

Predictive State:
  次に何が起きそうか
  何をするとどう変わるか
  どの情報が予測に必要か
```

知能一般に近いのは、未来予測と行動選択に役立つ圧縮表現である。

## 人間可読構造と潜在状態

`Entity`、`Relation`、`Belief`、`Claim` は人間にはわかりやすい。
しかし、世界モデルにとって最適な内部表現とは限らない。

```text
人間可読な状態:
  説明・監査・デバッグに便利

潜在状態:
  予測・制御・圧縮に便利
```

本体はlatent stateでよい。
人間可読な構造は、必要に応じて取り出す説明用ビューやキャッシュとして扱う。

## 外部メモリの位置づけ

外部メモリは世界モデルの本体ではない。

役割は次である。

```text
過去観測を保存する
必要な文脈を取り出す
長期履歴を圧縮する
モデルが再利用できる材料を渡す
```

```text
外部メモリ:
  観測履歴・キャッシュ・検索基盤

世界モデル:
  状態を作り、未来や行動結果を予測する本体
```

したがって、外部メモリの詳細よりも、モデルに何を見せるべきか、どの情報と関係に計算を使うべきかが本質である。

## Transformerの位置づけ

Transformerは自然言語モデルとしてだけでなく、トークン化された観測・行動・状態の関係を計算するエンジンとして使える。

```text
observation tokens
action tokens
state tokens
memory tokens
goal tokens
```

ただし、すべての関係を毎回見ると `N^2` で重い。
そのため、トークン粒度、関係選択、圧縮、検索が重要になる。

外部メモリやContext Builderは、世界モデルに渡す情報を選ぶための補助装置である。

## 自然言語の位置づけ

自然言語は世界モデルの入力の一種である。
ただし非常に特殊で強力な入力である。

```text
自然言語:
  人間が世界を抽象化して圧縮した観測
```

自然言語を捨てる必要はない。
むしろ、自然言語資産、世界モデル、非言語観測・行動・環境を接続することが重要である。

```text
自然言語資産
  ↔
世界モデル
  ↔
非言語観測・行動・環境
```

## 世界モデルの最小構成

```text
Observation Encoder:
  観測を表現に変換する

Action Encoder:
  行動を表現に変換する

Latent State:
  現在の推定状態

Transition Model:
  状態がどう変わるかを予測する

Prediction Head:
  次の観測、報酬、結果、リスクなどを予測する

Context / Memory Interface:
  必要な過去情報だけ取り出す

Language Interface:
  人間の自然言語と内部状態を接続する
```

## 忘れやすい論点

### 学習目的

世界モデルでは、表現の良し悪しは未来予測や行動選択に役立つかで決まる。

```text
次の観測を予測する
状態変化を予測する
行動結果を予測する
リスクを予測する
報酬を予測する
不確実性を予測する
```

### 行動

世界モデルは、世界を理解するだけでは不十分である。

```text
この行動をしたら何が起きるか
何をすれば望ましい状態に近づくか
何をすれば情報が増えるか
```

predictionだけでなくcontrolやplanningにつながる必要がある。

### 能動的観測

AIは入力を受け取るだけではなく、何を観測すべきかを選ぶ。

```text
検索する
質問する
実験する
ツールを呼ぶ
画像の別部分を見る
ロボットを少し動かして確認する
```

知能は、不確実性を減らすために観測を選ぶシステムでもある。

### 不確実性と探索

世界モデルは一つの未来だけではなく、複数の可能な未来を扱う必要がある。

```text
どの未来がどの程度ありそうか
どこが不確実か
どの行動で不確実性を減らせるか
```

### 報酬・価値・目的

世界がどう変わるかを予測するだけでは足りない。
どの状態が望ましいか、どの失敗が大きいか、どの行動がコストに見合うかが必要である。

ただし、価値や目的も細かく手設計しすぎるのは危険である。
タスク、フィードバック、環境との相互作用から学習できる形が望ましい。

### 時間スケール

状態には複数の時間スケールがある。

```text
瞬間的な感覚状態
短期の作業状態
会話やタスクの状態
長期記憶
世界の安定構造
モデル重みに入る一般知識
```

これらを同じStateとして扱うと混乱する。

### 階層性

世界モデルは単一粒度では厳しい。

```text
低レベル:
  センサー、ピクセル、音声フレーム、運動

中レベル:
  物体、イベント、行動、場所

高レベル:
  目的、計画、因果、社会的意味、タスク
```

低レベルから高レベルへの圧縮、必要時の再展開、階層間の通信をどう学習させるかが本題になる。

### 因果と反実仮想

世界モデルが強くなるには、相関だけでなく、介入と反実仮想を扱う必要がある。

```text
AをしたらBが起きる
AをしなかったらBは起きなかったか
別の行動ならどうなったか
```

### データの取り方

世界モデルでは、どんなデータで学習するかが難しい。

```text
動画
行動ログ
シミュレーション
ロボット実データ
人間フィードバック
ツール実行履歴
```

受動的な観測だけでは行動結果を学びにくい。
action-conditioned dataが重要になる。

### オンライン更新と非定常性

世界、ユーザーの目的、環境は変わる。

更新先を分ける必要がある。

```text
モデル重み
短期状態
長期メモリ
キャッシュ
文脈
```

毎回モデル重みを更新するのは重く危険である。

## 評価

世界モデルとして評価するなら、次を見る。

```text
次状態予測ができるか
行動結果を予測できるか
長期依存を扱えるか
矛盾する観測から仮説を更新できるか
少ない観測で適応できるか
未知環境に汎化できるか
自然言語知識を非言語行動に接続できるか
```

## まとめ

このプロジェクトの中心は、外部メモリを作ることではない。

```text
観測と行動から、予測に役立つ内部状態を作る。
その内部状態を、自然言語、マルチモーダル観測、ツール実行結果と接続する。
Transformerは、その関係計算のための汎用エンジンとして使う。
```

短く言えば、世界モデルを中心にし、外部メモリや自然言語はその入力、補助、インターフェースとして扱う。
