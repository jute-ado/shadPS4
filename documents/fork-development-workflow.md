# Fork development and promotion workflow

This document defines how experimental work in this fork moves into its public,
upstream-like branch. It is intentionally repository-generic: private games,
saves, routes, captures, machine maps, and raw logs remain outside Git.

## Branch roles

### `main`

`main` is the stable public branch. It should remain close to upstream and
contain only:

- portable emulator fixes that are not already implemented better upstream;
- focused automated tests protecting each retained fix;
- minimal user-facing documentation and the fork disclosure in `README.md`;
- changes that passed the complete unit and platform-scoped game-test gates.

`main` is not the place for investigation journals, temporary instrumentation,
private Test Lab inputs, or unfinished compatibility experiments.

### `dev`

`dev` is the public integration branch. It may contain sanitized investigation
documentation, generic Test Lab adapters, diagnostic schemas, and experimental
compatibility work. It must still build, and changes should still begin with a
focused failing test, but it may carry work that has not earned promotion to
`main`.

### Feature and promotion branches

- Begin investigative changes on a focused branch from `dev`.
- Merge successful investigation branches back into `dev` after their focused
  tests and applicable game tests pass.
- Promote a proven change through a fresh branch from `main`.
- Cherry-pick, squash, or reimplement only the smallest portable fix and the
  tests needed to protect it.

Do not merge `dev` wholesale into `main`. That would import experimental
ancestry and development-only material into the clean branch. Merge `main`
back into `dev` regularly instead.

## Upstream review

Before promoting a fork change:

1. Refresh the read-only upstream remote.
2. Inspect upstream changes touching the same behavior and ownership model.
3. Drop the fork patch if upstream already solves the problem.
4. Prefer upstream's implementation when it is stronger, then adapt the
   fork-specific test or Test Lab hook to that implementation.
5. If the fork remains necessary, port the minimal behavior without unrelated
   diagnostics or game-specific assumptions.

Patch-ID comparison is only an initial screen. A patch that is not byte-for-byte
equivalent may still be semantically obsolete because upstream solved the same
problem differently.

## TDD and validation gates

Every behavior change starts with a focused RED reproducing a generic invariant.
Promotion requires, in order:

1. focused RED and minimal GREEN;
2. affected component tests;
3. the complete unit suite;
4. the relevant platform-scoped Test Lab suite on the exact candidate binary;
5. NVIDIA validation for Uncharted compatibility changes;
6. AMD validation when the change is expected to be vendor-neutral or touches
   Vulkan/device behavior;
7. comparison with an accepted control where timing, capture, or infrastructure
   can affect the result;
8. a clean diff, privacy scan, and source/build provenance check.

A unit-green result is not sufficient for renderer lifetime, synchronization,
or ownership changes. A game-test failure must not be dismissed as
infrastructure without a valid control and complete evidence.

## Test Lab boundary

The repository may contain generic protocol code, adapters, schemas, synthetic
fixtures, and documentation needed to run Test Lab. It must not contain:

- copyrighted game data or executables;
- user saves, emulator user data, or memory-state snapshots;
- machine-specific emulator or asset maps;
- private input routes or manifests;
- screenshots, videos, RenderDoc captures, crash dumps, or raw game logs unless
  they have been deliberately reviewed and approved as public assets;
- credentials, tokens, personal filesystem paths, or private network details.

Public reports should retain only bounded, sanitized evidence needed to explain
the invariant and reproduce the test with separately held private inputs.

## Promotion inventory checkpoint

At the 2026-08-18 checkpoint, patch-ID screening found 103 fork-unique,
non-merge commits relative to upstream `11a5e47a`. None was patch-equivalent to
upstream. This does **not** mean all 103 changes are still needed: upstream may
have solved the same behavior with a different implementation.

The initial review queues are:

- **Keep on `dev`:** Test Lab capability negotiation, controller route
  recording/replay, automated capture integration, extensive investigation
  documentation, and diagnostic-only reporting.
- **Review for promotion:** generic fix-and-test pairs covering address-space
  mapping, DMA/BDA residency, sparse memory, buffer publication, Vulkan
  pipeline binding, presentation ownership, MUBUF `addr64`, texture subresource
  containment, video-decoder ownership, and Windows path/runtime behavior.
- **Require game-gate evidence:** Uncharted-derived changes whose tests express
  a generic invariant but whose only live validation is currently game-specific.
- **Hold:** the combined upstream-integration branch until its recorded AMD U1
  assertion regression is resolved and the complete current gates pass.
- **Do not promote:** historical result logs and investigation-only documents;
  retain their sanitized forms on `dev` instead.

Each review-for-promotion item needs an explicit upstream semantic comparison,
not merely a cherry-pick attempt. The candidate must be rebuilt from `main`,
carry its focused tests, and pass the gates above before it can be merged.

## Current restructuring checkpoint

The working public branch is preserved by the
`archive/public-main-before-cleanup-20260818` tag. The attempted reconstruction
from upstream was rejected as a promotion strategy because it discarded
working compatibility behavior and would require replaying a large, coupled
history without preserving its validation context.

The retained `main` and `dev` branches intentionally share the working
implementation. Their difference is documentation and development surface:
`main` carries the public fork disclosure, working code, and tests, while
`dev` additionally carries this knowledge base, investigation history, and
generic Test Lab guidance. Future functional changes are developed from
`dev`, validated, then promoted as focused code-and-test changes rather than by
merging `dev` wholesale.

The old `clean-main` branch is an archive/comparison only. It must not replace
the working public branch without independently recovering and validating full
functional parity.
