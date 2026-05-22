# Model Artifacts

Model artifacts are split into type definitions and concrete checkpoint entries.

## Type Definitions

Type definitions live in source code:

```text
src/intrep/representation/assembly_specs/
```

An assembly spec defines the model type:

- which input module is used
- which core module is used
- which output modules are used
- which hidden layout and output space the type expects
- which compatibility rules make a checkpoint loadable

Assembly specs do not point at checkpoint files and do not depend on `models/`.

## Component Implementations

Component implementations live in source code:

```text
src/intrep/representation/inputs/
src/intrep/representation/cores/
src/intrep/representation/outputs/
src/intrep/representation/assemblies/
```

Inputs, cores, and outputs implement reusable blocks. Assemblies read an
assembly spec and construct a model instance from those blocks.

## Checkpoint Entries

Concrete checkpoint entries live under `models/` when they are promoted as
loadable artifacts:

```text
models/<model-entry>/
  manifest.json
  components/
    input.pt
    core.pt
    policy_output.pt
    value_output.pt
```

`manifest.json` is an instance manifest. It records:

- checkpoint identity
- assembly id and assembly spec id
- input schema id and input feature manifest hash
- component file paths and hashes
- component module ids
- model dimensions needed to instantiate the assembly

It does not define the assembly type. The type is resolved from
`assembly_spec_id` through the source registry.

## Dependency Direction

The dependency direction is:

```text
models/<entry>/manifest.json
  -> checkpoint loader
  -> assembly_specs
  -> assemblies
  -> inputs / cores / outputs
```

The reverse direction is not allowed. Input, core, output, and assembly spec
modules must not depend on concrete checkpoint entries under `models/`.
