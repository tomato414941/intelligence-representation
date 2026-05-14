# Project Agent Policy

## Language
- Respond in Japanese.
- Write code comments in English.

## Commit Policy
- This project is intended to be publishable on GitHub.
- Do not commit secrets, local environment files, or personal machine paths.
- Use small commits with messages in the form `type: description`.

## Project Scope
- This repository explores representations for intelligence and meaning.
- Keep notes, experiments, and implementation artifacts separated.
- Avoid adding broad abstractions before there is a concrete experiment or repeated pattern.
- Ask for explicit approval before keeping backward compatibility paths, aliases,
  or deprecated interfaces.
- ShogiGameRecord schema is mirrored with `../shogi-arena-agent`; update both
  repositories' read/write/tests together when changing it.
- Computer-shogi self-play defaults should use `max-plies=320`; shorter
  overrides are allowed only with a warning.

## Verification
- Run relevant tests or checks after implementation changes.
- For note-only changes, review formatting and repository status.

## Compute Cost Notes
- Use `docs/compute-costs.md` to support expensive or remote run decisions, not
  as a general run log.
- Keep `docs/datasets.md` aligned when data-source or dataset layout changes.

## Training Model Size
- Use `d256-h1024-heads8-l6` for training runs unless there is a specific
  reason to use a smaller or larger model.
- Tests and smoke checks may use smaller models when model quality is not being
  evaluated.

## Run Artifacts
- Treat `runs/` as disposable experiment output that may be deleted at any time.
- Do not make `runs/` the canonical home for models or datasets that must be
  kept.
- Promote any checkpoint or tokenizer that must survive run cleanup into an explicit
  non-`runs/` location before depending on it.

## Dependencies
- PyTorch is an optional dependency so RunPod official PyTorch templates can use
  their system CUDA-compatible torch instead of replacing it from the project
  environment.
- Local and CI test environments must install the torch and vision extras before
  running the full unit test suite: `./scripts/setup_local.sh` or
  `uv sync --extra torch --extra vision`.
- RunPod setup must use `./scripts/setup_runpod.sh`. It intentionally avoids
  `uv sync` and installs the project without dependencies so the template's
  system PyTorch/CUDA stack is not replaced.
- RunPod torchvision setup must use `./scripts/setup_runpod_vision.sh` after
  `./scripts/setup_runpod.sh`. It keeps system torch intact and requires an
  explicit torchvision wheel spec for the selected RunPod image.
- Project-specific RunPod image, region, memory, and measurement notes live in
  `docs/runpod.md`.
