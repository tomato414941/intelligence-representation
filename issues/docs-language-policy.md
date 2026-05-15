# Docs Language Policy

Status: open
Priority: medium

## Problem

Several project documents contain Japanese prose. Current examples include:

- `docs/concept.md`
- `docs/predictive-representation-system.md`
- `docs/world-model.md`
- `docs/bitter-lesson.md`
- `docs/evaluation.md`

This is inconvenient for a public GitHub project if the intended durable
documentation language is English. It also makes the documentation style
inconsistent: many operational and experiment documents are already written in
English, while some conceptual documents remain Japanese.

## Desired Shape

The project should define a simple language policy for committed documentation.

Likely policy:

- Durable project docs are written in English.
- Japanese discussion can remain in conversation, local notes, or draft material
  until promoted.
- Existing Japanese docs are translated or rewritten in English before being
  treated as current project documentation.

The task is not to preserve a literal translation at all costs. Some conceptual
docs may be better rewritten into shorter English documents if the current text
is too broad or historical.

## Close Condition

- The intended language for durable project docs is documented.
- Current Japanese project docs are either translated, rewritten, moved out of
  current docs, or explicitly marked as drafts.
- `rg "[ぁ-んァ-ン一-龯]" docs issues README.md AGENTS.md` no longer finds
  unintentional Japanese prose in durable docs.
