# Long-Lived Artifact Home

Status: closed. Priority: low.

## Issue

Large long-lived artifacts may eventually outgrow git and local disk, but the
project has not chosen an external artifact home.

## Why It Matters

Datasets, caches, checkpoints, and evaluation bundles can become inputs for
future work. Local-only storage is fragile, while putting large generated files
in GitHub is not appropriate.

## Scope

This is intentionally deferred for now. No current artifact requires an external
artifact home.

- Decide when an artifact is too large or too important for local-only storage.
- Compare simple external homes such as Hugging Face Hub or object storage when
  there is a concrete artifact to preserve.
- Keep GitHub focused on code, docs, and small metadata.

## Non-Goals

- Do not introduce external storage before a real artifact needs it.
- Do not migrate existing local data immediately.
- Do not build a broad artifact registry.

## Acceptance Criteria

This issue can close when the project either chooses a long-lived artifact home
or explicitly defers that choice until a concrete artifact crosses a defined
threshold.

## Closure

Closed on 2026-05-24.

Use Cloudflare R2 as the project's formal external storage home for long-lived
large artifacts. Git remains for code, docs, and small metadata. Local disk and
`runs/` remain disposable working locations, not durable artifact homes.

R2 is the durable home for artifacts such as:

- tensor caches
- promoted checkpoints
- datasets or dataset-derived bundles that are too large for Git
- evaluation bundles that must survive local cleanup

Keep artifact semantics in project manifests and docs. R2 is the storage layer,
not a reason to couple project code to Cloudflare-specific APIs.
