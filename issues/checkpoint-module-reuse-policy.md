# Checkpoint Module Reuse Policy

Status: open.

## Issue

One of the project's main goals is to reuse input modules, shared cores, and
output heads across different problems.

There are two possible long-term directions:

- save reusable modules as separate artifacts
- save a larger checkpoint that contains many input modules, cores, and output
  heads together

Both may become useful, but choosing either storage format too early would add
management cost.

The current middle path is to save whole problem model checkpoints, while making
reuse happen explicitly at the module level.

Today, `load_compatible_shared_state()` initializes a model from another model
state by loading keys that have the same name and shape. That makes the reuse
boundary implicit. A future model could accidentally reuse a key name and shape
for a problem-specific head or adapter, and that component would be initialized
even if the transfer was intended to reuse only shared parts.

## Desired Direction

Keep checkpoint files simple for now:

- save whole problem model checkpoints
- do not split input/core/head files yet
- do not introduce a universal multi-interface checkpoint yet

Make reuse explicit:

- full checkpoint restore loads the whole model strictly
- transfer initialization selects named modules such as `core`,
  `image_input_layer`, `text_input_layer`, or `token_output`
- the implementation may use PyTorch state dict key prefixes internally, but the
  public API should speak in module names rather than raw prefixes

Keep full checkpoint restore strict.

For transfer initialization, prefer an API shaped like:

```python
module_names=("core", "image_input_layer")
```

This would keep `--init-checkpoint-path` as initialization, not compatibility,
while making it clear which modules may be reused.

## Acceptance Criteria

- full checkpoint loading remains strict
- transfer initialization does not silently load arbitrary matching keys
- reusable modules are selected by module name at the call site
- the project has not prematurely committed to separate module artifacts or one
  universal multi-interface checkpoint
