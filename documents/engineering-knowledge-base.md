# Engineering knowledge base

This index is the durable, public-safe engineering memory for the fork's
`dev` branch. It records reusable facts about PlayStation 4 behavior, shadPS4,
Vulkan, GPU drivers, compatibility investigations, and validation methods.
The goal is to help human and AI contributors continue an investigation
without repeating disproven experiments.

The knowledge base contains reduced technical findings, not private evidence.
Games, saves, controller routes, screenshots, videos, GPU captures, memory
images, raw logs, machine maps, credentials, private network details, and
machine-local paths must remain outside Git.

## Current guides and investigations

- [Fork development and promotion workflow](fork-development-workflow.md)
- [Emulator Test Lab workflow](emulator-test-lab.md)
- [Uncharted compatibility investigation guide](uncharted-compatibility-investigation.md)
- [Uncharted 1 cave hair investigation](u1-hair-investigation.md)
- [Investigation document template](investigation-template.md)

## What belongs here

- PS4, GNM/GNMX, kernel, ABI, shader, memory, and synchronization behavior.
- Translation decisions between guest AMD GPU state and Vulkan.
- Nvidia, AMD, Intel, operating-system, and driver differences that have been
  demonstrated with suitable controls.
- Renderer, shader compiler, texture cache, video decoder, input, audio, and
  presentation ownership models.
- Confirmed fixes, rejected hypotheses, ambiguous evidence, and unresolved
  questions.
- Synthetic tests protecting each confirmed invariant.
- Reproducible validation methodology expressed without private inputs.

## Evidence vocabulary

Every substantial claim should use one of these states:

- **Confirmed:** demonstrated by a synthetic test or authoritative source and
  validated against the relevant live control.
- **Probable:** supported by repeatable evidence but missing a decisive test or
  control.
- **Hypothesis:** plausible and testable, but not yet supported strongly enough
  to guide a behavior change.
- **Disproven:** contradicted by a controlled experiment.
- **Historical:** accurately describes an older revision but is not a claim
  about the current branch.
- **Superseded:** replaced by a newer document or implementation; retain a link
  to the replacement.

Code and tests are authoritative when documentation disagrees. Each document
must identify the revision and date on which its conclusions were last
verified. Re-check driver-sensitive conclusions after a driver, Vulkan feature,
operating-system, or adapter-selection change.

## Maintenance rules

1. Start new investigations from `documents/investigation-template.md`.
2. Separate observed facts from interpretations and proposed fixes.
3. Record negative results; they prevent expensive repetition.
4. Link the focused tests and commits that protect a conclusion.
5. Summarize private runs with bounded, non-identifying facts only.
6. Mark stale findings instead of silently rewriting history.
7. Merge completed implementation and sanitized findings into `dev`.
8. Promote only the minimal implementation, essential tests, and concise
   user-facing conclusion to `main`.
9. Merge `main` back into `dev` regularly; never merge `dev` wholesale into
   `main`.

