# Experiment 02: Contextual Claims

## 目的

人生スケールの意味記憶に向けて、Claimに時間、文脈、信念主体、矛盾を持たせる。

Experiment 01では、現在有効なClaimを単純に追加・無効化した。
Experiment 02では、矛盾を失敗として扱わず、意味状態の一部として保持する。

## 追加する概念

```text
time:
  いつ観測されたか、いつ成り立つ主張か

context:
  どの状況、場面、話題、視点での主張か

owner_of_belief:
  世界としての主張か、誰かの信念か

status:
  active / inactive / contradicted / uncertain / superseded

conflicts_with:
  どのClaimと衝突しているか

supersedes:
  どのClaimを更新・置換したか
```

## 例

```text
obs_1:
  田中が本を持っている

obs_2:
  佐藤が本を持っている
```

同じ時点・同じ文脈・同じ本について、所有者が複数いるなら衝突候補になる。

```text
claim_1:
  has(田中, 本)
  status: active

claim_2:
  has(佐藤, 本)
  status: uncertain
  conflicts_with: claim_1
```

ただし、矛盾とは限らない。

```text
時点が違う
同じ「本」ではない
片方が誰かの信念である
文脈が違う
観測が誤っている
```

したがって、衝突したClaimは即削除せず、根拠、時点、文脈、信頼度と一緒に保持する。

## 成功条件

```text
矛盾候補を検出できる
衝突するClaimを削除せず保持できる
active / uncertain / contradicted などの状態を区別できる
source と time と context をたどれる
```

この実験は、単一の現在状態から、文脈付きの意味記憶へ進むための足場である。
