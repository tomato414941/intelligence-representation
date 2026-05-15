# Saved File Format Policy

Status: closed.

## Issue

Saved files previously used several payload-format naming styles:

- `schema`
- `schema_version`
- unqualified identifiers such as `shogi_training_data_bundle_v1`
- qualified identifiers such as `intrep.text_tokenizer.v1`

This made compatibility decisions less obvious as checkpoints, tokenizers,
tensor caches, Training Data Bundles, Experience Stores, and run metrics grew.

## Policy

File names identify the file's role. They do not carry format versions.

Examples:

- `checkpoint.pt`
- `tokenizer.json`
- `manifest.json`
- `metrics.json`
- `shogi-policy-value-tensors.pt`

Reusable or loadable saved files put their format identifier in the payload
under `schema_version`. Loaders must check that value before trusting the
payload.

Run-local metrics and summaries may also use `schema_version` when they are
machine-read. Human-only logs do not need a schema marker.

Schema version identifiers should be qualified enough to avoid collisions and
include an explicit version suffix, for example:

- `intrep.shogi_training_data_bundle.v1`
- `intrep.shogi_experience_store.v1`
- `intrep.shogi_policy_value_tensor_cache.v1`

## Acceptance Criteria

This issue can close when current saved-file payloads use `schema_version`
consistently and the file-name versus payload-format rule is documented.

Met.

## Resolution

Current saved-file payloads now use `schema_version` consistently. The file-name
versus payload-format rule is documented in `docs/artifact-layout.md`.

## Non-Goals

- preserve compatibility with every historical local saved file
- create a broad artifact registry
