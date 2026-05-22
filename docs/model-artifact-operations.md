# Model Artifact Operations

Manual component replacement is allowed for model entries.

After editing `models/<entry>/manifest.json` or files under
`models/<entry>/components/`, validate the entry:

```sh
uv run python -m intrep.problems.shogi_policy_value.validate_checkpoint models/<entry>
```

Do not use a manually edited model entry until validation passes.
