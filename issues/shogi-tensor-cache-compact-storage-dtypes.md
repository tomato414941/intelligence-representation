# Shogi Tensor Cache Compact Storage Dtypes

Status: open. Priority: high.

## Issue

The shogi policy/value tensor cache stores many durable ID tensors with runtime
training dtypes. In the current policy-plane cache, a 10,000-sample shard is
about 486 MB as `.pt`, and the full Qhapaq policy-plane cache projects to more
than 200 GB uncompressed.

The largest contributor is `square_feature_ids`, which is stored as `int64`.
These values are feature IDs with a small vocabulary, so they do not need
64-bit storage on disk. `nn.Embedding` needs `long` at use time, but the cache
does not need to persist them as `int64`.

## Desired Shape

Tensor caches should store compact durable dtypes and cast to runtime dtypes at
load/batch time.

Likely durable dtypes:

- feature IDs: `uint16` or `int16`
- action IDs and policy-plane labels: `uint16` or `int32`, depending on range
- pair relation IDs: `uint8`
- relation edge indices: `uint16`
- value targets and weights: keep `float32` unless a separate precision
  decision is made

The cache manifest should record storage dtypes so cache readers can validate
and cast intentionally.

## Non-Goals

- change the model input representation
- change policy/value targets
- quantize model weights
- make compressed cache loading mandatory

## Notes

Representative measurement from the Qhapaq policy-plane cache:

- 10,000-sample shard `.pt`: about 486 MB
- projected full cache uncompressed: about 247 GB
- same shard compressed with `zstd -3`: about 18 MB

Compression helps, but dtype compaction should happen first because the durable
cache format is currently wider than the represented values require.
