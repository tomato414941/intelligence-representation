# Transfer Initialization Prefix Policy

Status: open.

## Issue

`load_compatible_shared_state()` initializes a model from another model state by
loading keys that have the same name and shape.

That is convenient, but the transfer boundary is implicit. A future model could
accidentally reuse a key name and shape for a problem-specific head or adapter,
and that component would be initialized even if the transfer was intended to
reuse only shared parts.

## Desired Direction

Keep full checkpoint restore strict.

For transfer initialization, make the intended reusable parts explicit. A small
option is to require allowed prefixes such as:

```python
allowed_prefixes=("core.", "image_input_layer.")
```

This would keep `--init-checkpoint-path` as initialization, not compatibility,
while making it clear which modules may be reused.

## Acceptance Criteria

- transfer initialization does not silently load arbitrary matching keys
- allowed reusable module prefixes are visible at the call site
- full checkpoint loading remains strict
