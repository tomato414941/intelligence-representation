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

## Verification
- Run relevant tests or checks after implementation changes.
- For note-only changes, review formatting and repository status.

## Dependencies
- PyTorch is an optional dependency so RunPod official PyTorch templates can use
  their system CUDA-compatible torch instead of replacing it from the project
  environment.
- Local and CI test environments must install the torch extra before running the
  full unit test suite: `uv sync --extra torch`.
- RunPod setup should avoid installing the torch extra unless the selected image
  does not already provide a compatible PyTorch installation.
