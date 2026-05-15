# Artifact Schema Naming Policy

Status: open.

## Issue

Artifact payloads use several schema naming styles:

- `schema`
- `schema_version`
- names ending in `_v1`
- names without an explicit version suffix

This has not caused a concrete loading bug yet, but it makes new artifact
formats harder to name consistently. As checkpoints, tensor caches, run metrics,
training bundles, and stores grow, inconsistent schema naming will make
compatibility decisions less obvious.

## Desired Direction

Define a small project policy for artifact schema identifiers:

- which key to use for persistent loadable artifacts
- whether schema identifiers should include version suffixes
- when changing the schema identifier is required
- how run metrics differ from durable loadable artifacts

Keep the policy small. Do not retrofit every historical artifact unless the
current code actively benefits from the cleanup.

## Acceptance Criteria

This issue can close when new artifact formats have a documented schema naming
rule, and current code has either been aligned where worthwhile or explicitly
left unchanged as historical.

## Non-Goals

- migrate all existing artifacts
- preserve compatibility with every historical local artifact
- create a broad artifact registry
