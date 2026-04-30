# Local Project Override

## Working Directory
- Use `/home/dev/projects/intelligence-representation` as the project root.

## Local Artifacts
- Treat `data/`, `runs/`, and model checkpoints as local/generated artifacts unless the user explicitly asks to version a specific file.
- Do not commit downloaded datasets, generated images, run metrics, or checkpoint files by default.

## Development Guardrails
- Keep experimental paths explicitly marked as temporary when they are not intended as the main architecture.
- Prefer deleting or replacing temporary validation code once the main path exists.
- Avoid broad multitask or multimodal abstractions until at least two concrete tasks require the same interface.
- Keep changes small; this project is especially sensitive to code growth and terminology drift.

## Commit And Push
- Commit and push small verified changes at the agent's discretion.
- Keep commits narrow and use `type: description` messages.
- Do not commit generated data, run outputs, checkpoints, secrets, or unrelated user changes.
- Ask before committing broad rewrites, destructive changes, or changes with unresolved test failures.

## Verification
- For implementation changes, run the narrow relevant tests first.
- Run `uv run python -m unittest` before committing changes that touch shared model or training code.
