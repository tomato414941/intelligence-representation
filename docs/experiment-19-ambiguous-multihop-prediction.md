# Experiment 19: Ambiguous Multi-Hop Prediction

## 目的

Multi-hop predictionが一意に解けないケースを扱う。

Experiment 18では、関係をたどれば正解が一つに決まった。
Experiment 19では、途中で分岐する観測を入れ、無理に一つへ潰さず候補集合として保持する。

## 例

```text
obs_1:
  located_at(鍵, 箱)

obs_2:
  located_at(箱, 棚)

obs_3:
  located_at(箱, 机)

action:
  find(太郎, 鍵, unknown)

prediction:
  ambiguous
  candidates:
    located_at(鍵, 棚)
    located_at(鍵, 机)
```

## 評価

```text
prediction_state:
  unsupported | resolved | ambiguous

candidate_facts:
  可能な予測候補

expected_candidates:
  期待される候補集合
```

正解判定では、単一factではなく候補集合を比較する。

## 意味

この実験は、不確実性と複数仮説を扱う最小形である。

重要なのは次である。

```text
矛盾や分岐を即座に削除しない
新しい情報で古い情報を雑に上書きしない
一意に決まらないものは ambiguous として保持する
```

## 成功条件

```text
分岐するmulti-hopでprediction_stateがambiguousになる
候補factを複数保持できる
候補集合として正解判定できる
```
