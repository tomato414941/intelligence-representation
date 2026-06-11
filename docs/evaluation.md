# Evaluation

## 役割

この文書は、評価の考え方をまとめる。
実装状態、CLI、テスト一覧、実験ログの正本ではない。

評価で守るべき中心は、次である。

```text
training objective と project claim を分ける
smoke metric と evidence metric を分ける
平均 loss と能力獲得を同一視しない
```

## Project Claim

このプロジェクトの上位仮説は次である。

```text
A predictive representation system for language, perception, action, memory, and belief.
```

この仮説は広い。
したがって、単一の損失低下や単一タスクの accuracy だけでは、
Predictive Representation System が実現したとは言えない。

評価では、どの主張をしているのかを常に分ける。

```text
training works:
  optimizer, model, batching, data path が壊れていない

task works:
  特定タスクで有用な予測や分類ができる

world-modeling evidence:
  観測・行動・文脈の違いに応じて未来予測が変わる

predictive-representation-system evidence:
  複数の入力形式やタスクを共有予測計算へ接続できる
```

## Smoke Metrics

smoke metric は、学習経路が最低限動いているかを見るための指標である。

```text
training loss reduction
held-out loss reduction
perplexity reduction
small task accuracy above chance
ranking helper sanity checks
```

これらは重要だが、主張は狭い。
たとえば平均 loss が下がっても、それだけでは world model ができたとは言わない。

## Evidence Metrics

より強い主張には、主張に対応した評価が必要である。

### Text

テキストでは、単なる training loss だけでなく held-out continuation を見る。

```text
held-out loss
perplexity
continuation ranking
longer-context degradation
```

### Image

画像では、分類なら task accuracy が基本である。
ただし、画像を shared predictive model に接続したいなら、
分類 accuracy だけでなく、画像条件付きの continuation や選択肢 ranking も見る。

```text
classification accuracy
choice ranking accuracy
image-conditioned text or label continuation
```

### World Modeling

world-modeling 的な主張には、行動条件付きの未来予測が必要である。

```text
same observation, different action -> different predicted outcome
same action, different context -> different predicted outcome
held-out next-observation ranking
counterfactual or intervention-sensitive prediction
```

見るべきなのは、表面的な系列補完ではなく、
観測、行動、文脈の違いが予測に反映されるかである。

### Shared Core

共有中間層を主張するなら、単にコードが同じクラスを呼んでいるだけでは足りない。
少なくとも、複数の入力経路が同じ予測計算へ接続され、
それぞれのタスクで劣化や改善が測れる必要がある。

```text
text path works
image path works
shared core path works
input-layer-specific failures are separable from core failures
```

## Negative Results

失敗は捨てない。
ただし、何に失敗したのかを狭く読む。

```text
data scale problem
context length problem
task construction problem
evaluation leakage
model capacity problem
optimization problem
representation problem
```

1つの条件で失敗しても、構造的に不可能とは結論づけない。
逆に、1つの smoke metric が成功しても、大きな能力獲得を主張しない。

## Acceptance Criteria

変更は、少なくとも次のどれかを明確にするべきである。

```text
training path is working
held-out generalization is measured
task metric is measured
world-modeling claim has a matching future-prediction test
shared-core claim has a matching cross-path test
failure mode is easier to diagnose
```

避けるべきこと:

```text
metric を増やすだけで主張を明確にしない
平均 loss だけで大きな能力を主張する
実験ごとに新しい broad schema を作る
評価に必要ない ontology を先に足す
```

## Current Claims

最終更新: 2026-06-10。測定の数値と詳細は、各 issue と git 履歴を正本とする。
この節は「いま何を主張してよいか」だけを持つ。

現在支持される主張:

```text
- テキスト・画像・画像/テキストのタスクは小さな Transformer 学習経路に
  接続でき、小〜中規模のローカル実行で損失低下またはタスク指標の改善を示す
- GridWorld の held-out セル汎化の失敗は、絶対セル ID 出力形式と
  データ支持不足の複合として説明済み
  (issues/closed/grid-world-heldout-generalization.md)
- 規則を注入していない次観測予測モデルが、Life (6x6) のランダム盤面の
  予測練習だけから更新規則を獲得する。N=16 では暗記のみ、N=1024 で
  未見盤面のズル耐性スコア両方がほぼ 1.0(3 seed で確認)
  (issues/grid-next-observation-emergence.md)
```

まだ主張できないこと:

```text
- 汎用の予測表現システム、潜在世界モデル
- 規則を注入せずに学習だけで獲得された held-out 汎化
  (測定中: issues/grid-next-observation-emergence.md)
- 信念の更新、記憶の読み書き、プランニング、制御
- 大規模なマルチモーダル生成、頑健な画像条件付き自由記述生成
```
