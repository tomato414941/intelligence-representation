# Experiment 03: Contextual State Update

## 目的

Experiment 01の状態更新と、Experiment 02の文脈付きClaimを統合する。

入力はまだ自然言語ではなく、構造化イベントとする。
ただし、各イベントは `time`、`context`、`owner_of_belief` を持つ。

## 入力例

```text
t1:
  田中が本を持っている

t2:
  田中が佐藤に本を渡した
```

構造化イベントとしては次のように扱う。

```text
claim:
  subject: 田中
  predicate: has
  object: 本
  time: t1
  context: world

transfer:
  actor: 田中
  recipient: 佐藤
  object: 本
  time: t2
  context: world
```

## 更新ルール

```text
claim(subject, predicate, object):
  add predicate(subject, object)

transfer(actor, recipient, object):
  deactivate latest active has(actor, object)
  add has(recipient, object)

place(actor, object, location):
  deactivate latest active has(actor, object)
  add located_at(object, location)
```

`deactivate` ではClaimを削除しない。
過去のClaimは `inactive` にして、どのイベントで無効化されたかを保持する。

## 矛盾処理

同じ `time`、`context`、`owner_of_belief` において、同じ `object` を複数の `subject` が `has` している場合は衝突候補にする。

衝突したClaimは即削除せず、`uncertain` として保持する。

## 成功条件

```text
t1: has(田中, 本) が追加される
t2: transfer により has(田中, 本) が inactive になる
t2: has(佐藤, 本) が active になる

同時点・同文脈で別所有者の has が入った場合は uncertain になる
```
