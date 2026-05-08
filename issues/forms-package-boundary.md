# Forms Package Boundary

Status: open.

## Issue

`forms/` is a possible future source-side package group for form/input-oriented
code, but adding it too early would create a broad abstraction before it solves
a concrete package boundary problem.

The current source-side packages are intentionally split only where concrete
pressure exists:

- `vision/` and `text/` are form/input-oriented.
- `worlds/shogi/` and `worlds/grid/` are world-oriented.

`domain` should not be used as the umbrella term for these packages. The open
question here is when `vision/`, `text/`, and future form-oriented packages
should move under a clearer `forms/` boundary.

## Do Not Create `forms/` Just Because

Do not introduce `forms/` only because:

- the name looks architecturally cleaner
- `worlds/` exists
- a future audio, video, or tabular package might be added someday
- `vision/` and `text/` can both be described as input forms
- a document can define the abstraction but no active code decision depends on it

That would make the abstraction lead the implementation.

## Create `forms/` When It Solves A Current Problem

Creating `forms/` becomes appropriate when at least one of these is true:

- A new form/input-oriented package is being added, and the existing flat layout
  makes the placement unclear.
- `vision/` and `text/` need shared form/input-layer utilities that do not
  belong in either package or in `core/`.
- Existing code reviews or implementation decisions repeatedly confuse
  form/input-oriented packages with world-oriented packages.
- A concrete file move would make a current dependency boundary simpler, for
  example by separating loaders, format conversion, tokenization, or input
  layers from task objectives.
- A source record can intentionally be represented through multiple forms, and
  the code needs one place to express those form conversions without making a
  problem package or world package own them.
- Audio, video, tabular, rendered layout, or other non-text/non-image inputs are
  implemented, and their package placement would otherwise repeat the
  `vision/` / `text/` ambiguity.
- Documentation and code disagree about the package boundary in a way that
  affects active implementation, not only wording.

## Candidate Scope

If introduced, `forms/` should hold form/input-oriented source-side code such
as:

- text forms, tokenization, and token encodings
- image forms, image IO, and patch/input layers
- audio/video loaders and input layers, if added
- form conversion utilities that are not objective-specific

It should not hold:

- task objectives, losses, metrics, training loops, or checkpoints
- world transition logic, replay, action stepping, or experience records
- domain-specific game rules such as legal shogi moves
- shared Transformer core code

## Acceptance Criteria

This issue can close when either:

- `forms/` is introduced because one of the concrete conditions above is met,
  with moved files and imports matching the documented responsibility, or
- the project explicitly decides to keep form/input-oriented packages flat and
  documents why the flat layout is still clearer.
