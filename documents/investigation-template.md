# Investigation: <short descriptive title>

## Metadata

- **Status:** Hypothesis | Probable | Confirmed | Disproven | Historical | Superseded
- **Area:** <PS4 subsystem / emulator component / Vulkan / driver>
- **Started:** YYYY-MM-DD
- **Last verified:** YYYY-MM-DD
- **Verified revision:** `<commit>`
- **Adapters and drivers:** <public-safe vendor, model class, driver version>
- **Supersedes:** <document or none>
- **Superseded by:** <document or none>

## Problem

Describe the observable behavior without assuming its cause. State the smallest
reproduction boundary and distinguish emulator behavior from test
infrastructure behavior.

## Relevant architecture

Summarize the PS4 contract, emulator ownership model, Vulkan requirements, and
known vendor differences needed to understand the investigation. Link primary
specifications or source locations where possible.

## Confirmed facts

- Fact, with the test, source, or bounded evidence that establishes it.

## Hypotheses

| Hypothesis | Status | Discriminating test | Result |
| --- | --- | --- | --- |
| <claim> | Hypothesis | <control or synthetic RED> | <pending> |

## Experiments and controls

Record the invariant under test, relevant control, revision, adapter class,
and reduced result. Do not record private paths, asset names unnecessarily,
raw logs, images, addresses, identifiers, or credentials.

## Decision

Explain why the chosen implementation follows from the evidence. State which
more invasive alternatives were rejected and why.

## Tests and validation

- Focused RED/GREEN test:
- Component suite:
- Complete unit suite:
- Nvidia game gate, when applicable:
- AMD game gate, when applicable:
- Accepted control:

## Rejected approaches

Retain technically useful negative results and the conditions under which they
were rejected.

## Remaining unknowns

- Open question and the smallest next experiment that could answer it.

## Privacy review

- [ ] No game packages, extracted content, saves, screenshots, videos, GPU
      captures, memory images, or raw logs.
- [ ] No machine-local paths, maps, private routes, network details, tokens,
      credentials, or personal information.
- [ ] Private observations are reduced to bounded technical facts.

