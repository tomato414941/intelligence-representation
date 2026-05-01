# Experiment 07: World State Update

## 目的

StateTransitionをWorldStateに反映する。

Experiment 06では、`action_result` からEventとStateTransitionを作った。
Experiment 07では、StateTransitionの `before` を無効化し、`after` を現在状態として有効化する。

## 入力例

初期状態:

```text
has(佐藤, 本): active
```

観測:

```text
StateTransition:
  before:
    has: [佐藤, 本]
  after:
    located_at: [本, 図書館]
```

期待:

```text
has(佐藤, 本): inactive
located_at(本, 図書館): active
```

## WorldState

```text
WorldState:
  active facts
  inactive facts
  transition history
```

Factは最小限、次を持つ。

```text
subject
predicate
object
status
source
invalidated_by
```

## 成功条件

```text
before factをinactiveにできる
after factをactiveにできる
元factは削除せず履歴として残る
どのtransitionで無効化されたか追跡できる
```
