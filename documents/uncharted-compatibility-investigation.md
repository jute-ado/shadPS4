# Uncharted compatibility investigation guide

This document records the reusable engineering method and accepted findings from
the Uncharted compatibility campaign. It is intentionally free of games, saves,
captures, machine paths, private manifests, and raw logs. Those artifacts remain
in the local Test Lab evidence store.

The goal is not to accumulate title-specific workarounds. Each accepted change
must repair a general emulator invariant, begin with a synthetic failing test,
and pass PS4-only regression gates.

## Campaign status

| Area | Status | General invariant |
| --- | --- | --- |
| Video decode picture metadata | Accepted | Picture metadata remains owned until the consumer has finished with it. |
| Video decode packet data | Accepted | Packet storage outlives asynchronous decode and presentation work. |
| Buffer lookup | Independently audited | Selection and containment behavior is covered by focused tests and PS4 gates. |
| Uncharted 1 cave hair | Accepted | MUBUF `addr64` decoding and source-buffer residency are preserved. |
| Uncharted 3 pub flicker | Active | Cached shader permutation compilation must consume one coherent guest-resource generation. |

Compatibility claims are checkpoint-specific until the complete campaign
playthrough and regression matrix is green.

## Required development loop

1. Reproduce the defect at a deterministic checkpoint on clean accepted main.
2. Preserve the immutable run and classify process, marker, visual, temporal,
   audio, and performance health independently.
3. Localize the first broken emulator invariant. Do not infer a cause from the
   last subsystem that happened to log a warning.
4. Add a focused synthetic RED that fails for the missing contract.
5. Implement the smallest general GREEN without title, shader-hash, draw, or
   asset special cases.
6. Run the focused test, affected subsystem tests, Uncharted-focused PS4 suite,
   and complete PS4 regression suite.
7. Run repeated clean checkpoint trials and review the actual rendered frames
   or video manually.
8. Update this document with proven, rejected, and still-unknown findings.

Keep code, corpus metadata, and private evidence in separate repositories and
worktrees. See [Emulator Test Lab workflow](emulator-test-lab.md) for the
cross-repository rules.

## Visual acceptance is multidimensional

A temporal oracle answers whether frames change unexpectedly. It does not prove
that the frame is the correct scene. A completely wrong but stable image can
pass a temporal-only test.

Every visual fix therefore needs all of these gates:

- **Scene identity:** compare against a reviewed reference from the same
  checkpoint and camera state.
- **Static correctness:** use bounded full-frame difference and structural or
  cosine difference thresholds.
- **Temporal correctness:** check abrupt returns, localized returns, visibility,
  distinct-frame count, and adjacent-frame differences.
- **Manual review:** inspect the first frame, representative transitions, and a
  playable video of the full checkpoint.
- **Process health:** keep exit, timeout, signal, required-marker, and
  forbidden-marker results separate from image quality.

Do not widen a visual threshold to admit a candidate. If animation makes a
single static baseline unsuitable, improve the oracle or capture contract.

### Rejected U3 false positive

An experiment that changed immediate constant reads and stream-buffer watch
lifetime made the temporal test pass, but manual review showed a large dark
polygon covering most of the scene. The oracle had compared it to an unrelated
loading-state image with permissive static thresholds, so a stable corruption
was incorrectly classified as a fix.

That experiment is rejected. It must not be merged or used as evidence that U3
is fixed. The corrected oracle uses a reviewed full pub-scene reference and
rejects the stable polygon by both mean absolute and cosine difference.

The lesson is general: a temporal pass is only meaningful after scene identity
and static correctness are established.

## Current U3 failure signature

Clean accepted main renders the intended pub scene, but adjacent frames can
alternate between:

1. a complete frame;
2. a frame with large rectangular regions of geometry or HUD missing or dark;
3. a return to the complete frame.

This is an A-B-A corruption pattern in an otherwise correct scene. A candidate
that replaces it with a stable wrong scene is a regression, even if every
temporal metric improves.

Current evidence rules out several sampled-image, sampler, cube-coordinate, and
content-lineage hypotheses as unique discriminators. These diagnostics remain
useful negative evidence, but they do not authorize renderer behavior changes.

## Cached shader permutation generation

The strongest scene-preserving lead is a generation-coherence invariant in the
shader cache. When an existing shader program needs a new specialization:

1. the cached `Info` is refreshed from current user data;
2. specialization parses flattened descriptors and the fetch shader;
3. a fresh `Info` is constructed to compile the new permutation;
4. translation may otherwise walk mutable guest descriptors and fetch code a
   second time.

If guest memory changes between steps 2 and 4, the specialization key can
describe generation A while the compiled module consumes generation B. The
module is then cached under a key that does not match its compiled resources.

The candidate contract is deliberately narrow:

- capture the exact flattened descriptor image after the existing refresh;
- retain the parsed fetch-shader result used by specialization;
- pass both snapshots only through the synchronous new-permutation compile;
- replay them before translation can read live guest memory again;
- keep the snapshots transient and out of serialized shader metadata;
- preserve the normal live path for initial compilation and cache hits.

This contract requires focused RED tests for generation replay, malformed
snapshot rejection, fetch replay, call ordering, and transient ownership. It is
not accepted until the corrected full-scene oracle and broader PS4 gates pass.

## Evidence classifications

Use precise language in reviews and handoffs:

- **Accepted:** synthetic RED/GREEN, clean checkpoint repetition, focused PS4
  suite, full PS4 regression, and manual visual review all pass.
- **Candidate:** tests pass and the checkpoint improves, but one or more required
  gates remain.
- **Negative evidence:** a measured boundary is uniform across positives and
  controls, so it does not authorize a behavior change.
- **Rejected:** the candidate regresses correctness, relies on a title special
  case, or passed an invalid oracle.
- **Unexercised:** the observed route did not use the hypothesized behavior.

Never convert correlation, timing changes, or a reduced event count into a
causal fix claim without a discriminating invariant.

## Privacy and publication boundaries

Safe to commit:

- general invariants and architecture descriptions;
- synthetic tests and fixtures;
- portable scenario contracts that contain no private assets;
- aggregate pass/fail and bounded performance summaries;
- public commit identifiers.

Keep private and outside Git:

- commercial game files and extracted shaders;
- saves, controller recordings, screenshots, videos, GPU captures, and memory
  dumps;
- machine maps, absolute local paths, credentials, and private service URLs;
- raw logs that may contain paths, identities, or guest content.

## Merge checklist

- [ ] The defect reproduces on clean accepted main with a reviewed full-scene
      reference.
- [ ] A focused synthetic test is RED before production edits.
- [ ] The GREEN repairs a general invariant without a title or shader special
      case.
- [ ] Focused tests and the affected subsystem test target pass.
- [ ] At least three fresh checkpoint trials pass static, temporal, process, and
      manual-review gates.
- [ ] The Uncharted-focused PS4 suite passes.
- [ ] The complete PS4 regression suite passes.
- [ ] Audio and performance results do not regress.
- [ ] Emulator and corpus revisions are recorded together when test intent
      changes.
- [ ] Documentation distinguishes accepted, rejected, negative, and unknown
      findings.
- [ ] Only source, tests, and sanitized Markdown are committed; private evidence
      remains external.
