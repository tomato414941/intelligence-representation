# Synthetic vision transition dataset boundary

## Issue

`src/intrep/domains/vision/synthetic_transitions.py` defines a concrete moving-dot
transition dataset with actions and next frames.

That source is useful, but it is not an image-domain primitive. It is a concrete
synthetic visual transition dataset.

## Desired Shape

- `domains/vision` keeps image-general concepts and image format utilities.
- Concrete synthetic visual datasets live under `datasets/vision`.
- Future visual prediction problems can depend on the dataset source without
  making it part of the vision domain definition.

## Scope

- Move moving-dot transition data generation out of `domains/vision`.
- Keep it independent from any specific problem.
- Update tests and imports.
