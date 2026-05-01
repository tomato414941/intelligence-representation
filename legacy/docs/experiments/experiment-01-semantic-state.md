# Experiment 01: Semantic State

## 目的

自然文の履歴ではなく、更新可能な意味状態として情報を保持できるかを確認する。

この実験ではLLMを使わない。
入力は人間が構造化したイベント列とし、更新ルールも人間が書く。

## 入力

```text
田中が本を持っている
田中が佐藤に本を渡した
佐藤が本を図書館に置いた
```

この自然文を、最初は手で次のようなイベントに変換する。

```text
claim:
  subject: 田中
  predicate: has
  object: 本

transfer:
  actor: 田中
  recipient: 佐藤
  object: 本

place:
  actor: 佐藤
  object: 本
  location: 図書館
```

## データモデル

```text
Observation:
  入力イベントの根拠

Claim:
  現在または過去に成り立つとされた関係

StateUpdate:
  Claimの追加と無効化の差分

SemanticState:
  現在有効なClaimと履歴
```

## 更新ルール

```text
claim(subject, predicate, object):
  add predicate(subject, object)

transfer(actor, recipient, object):
  deactivate has(actor, object)
  add has(recipient, object)

place(actor, object, location):
  deactivate has(actor, object)
  add located_at(object, location)
```

古い主張は削除せず、`active: false` として履歴に残す。

## 成功条件

最後の状態が次のようになれば成功とする。

```text
has(田中, 本): inactive
has(佐藤, 本): inactive
located_at(本, 図書館): active
```

この実験は、意味を「入力文の保存」ではなく「状態更新」として扱う最小例である。
