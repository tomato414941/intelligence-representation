# Experience Store Dataset Boundary

Status: open.

## Issue

The project does not clearly define whether an Experience Store is a Dataset.

Current shogi data can look ambiguous because both Experience Store and
Training View directories may contain `games.jsonl`, and both contain data that
could eventually be used for training.

The intended distinction should be made explicit:

- Experience Store: durable source storage for generated or collected
  experience
- Data Selection: a declared-use data inclusion boundary
- Training View: a fixed source snapshot used by Data Selection or training

Under this interpretation, an Experience Store is source material for datasets,
not a dataset by itself.

## Why It Matters

If Experience Store is treated as a Dataset, training can accidentally depend
on "whatever is currently in the store." That makes train/eval boundaries,
target source selection, source mix, and reproducibility unclear.

If Experience Store is treated only as source storage, training must go through
explicit Data Selection or a fixed Training View. That keeps the learning
contract clearer even when generated experience grows over time.

This also affects non-shogi data:

- external datasets such as MNIST or Qhapaq may start as source data
- generated interaction data may start in an Experience Store
- both should become trainable only through explicit Data Selection or an
  equivalent learning contract

## Acceptance Criteria

This issue can close when the project defines:

- whether Experience Store is a Dataset or only a source store
- how Experience Store differs from external dataset source data
- whether training is allowed to read directly from an Experience Store
- which artifact owns train/eval split and target derivation

The decision should be reflected in the relevant docs or code names.

## Non-Goals

- rename Experience Store immediately
- redesign directory layout
- implement Replay Buffer or sampling policy
- change ShogiGameRecord schema
