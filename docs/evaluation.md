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

最終更新: 2026-09-10。測定の数値と詳細は、各実験文書、issue と git 履歴を正本とする。
この節は「いま何を主張してよいか」だけを持つ。

現在支持される主張:

```text
- 言葉・画像・音・行動・実際の結果を、同じ Transformer と学習される再帰的記憶に
  接続し、行動・文章・次観測の予測を同時に学習できる。小さな移動環境の未学習
  128配置では、教師行動との一致79.17%、目標の色への返答100%。言葉・画像・音を
  それぞれ伏せると行動一致は61.46%・56.77%・53.52%、記憶を毎回初期化すると63.28%。
  実行した経験の保存と混合 replay による再学習も動作するが、その短い1巡での
  性能向上は未成立。画像予測は直前画像をコピーする基準より弱い。
  (docs/multimodal-agent.md)
- テキスト・画像・画像/テキストのタスクは小さな Transformer 学習経路に
  接続でき、小〜中規模のローカル実行で損失低下またはタスク指標の改善を示す
- GridWorld の held-out セル汎化の失敗は、絶対セル ID 出力形式と
  データ支持不足の複合として説明済み
  (issues/closed/grid-world-heldout-generalization.md)
- 規則を注入していない次観測予測モデルが、Life (6x6) のランダム盤面の
  予測練習だけから更新規則を獲得する。N=16 では暗記のみ、N=1024 で
  未見盤面のズル耐性スコア両方がほぼ 1.0(3 seed で確認)
  (issues/closed/grid-next-observation-emergence.md)
- 複数ルールで学習したモデルが、観測した変化前後の例だけから、同じ
  セルオートマトン族の未学習ルールに適応できる。6x6盤面・未学習64ルール・
  学習seed 31では、例0組の54.09%から8組の98.86%へ次状態予測が改善。
  別ルールの例への差し替え、空間配置を無視する頻度予測との比較も実施。
  (docs/research/cellular-rule-inference.md)
- 上記の8組の精度は、学習seed 31/32/33で98.63〜98.86%と再現。
  観測出力にノイズを含めて学習すると、別の最終評価の10%ノイズ条件で
  95.90%から98.00%へ改善。さらにルール変更を含めると、旧4組・新4組・
  10%ノイズで93.02%に達するが、安定時の精度は低下する。
  拡張モデルは各1 seed。観測順の利用を支持するが、最適な変化検出や
  固定窓への一律の優位性、永続的な記憶の獲得を示す結果ではない。
  (docs/research/cellular-rule-stress.md)
- 既存の固定モデルによる結果予測を、37通りの操作の一手先選択に利用できる。
  未学習64世界で、実行した8回分の変化を文脈として使うと、次の新しい盤面・
  目標で最良の操作を選ぶ割合は学習seed 31/32/33で91.76〜94.12%。
  同じ課題で文脈を空にすると7.06〜13.73%。操作間で得点が異なる255課題で評価。
  操作の列挙、目標効用、履歴保持は手実装であり、学習されたプランニングや
  永続的な記憶の証拠ではない。
  (docs/research/cellular-rule-control.md)
```

まだ主張できないこと:

```text
- 汎用の予測表現システム、潜在世界モデル
- 未経験のルール族・盤面サイズへの汎化、部分観測からの規則推測
- ノイズ・ルール変更を含む拡張学習の効果が複数の学習seedで安定して再現すること
- 信念更新の妥当性、長期にわたる記憶の保持、学習されたプランニング、複数手先の制御
- 自分で集めた経験の replay が、同じ計算量の既存データ学習より能力を高めること
- 大規模なマルチモーダル生成、頑健な画像条件付き自由記述生成
```
